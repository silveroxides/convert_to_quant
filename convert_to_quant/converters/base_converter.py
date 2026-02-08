"""
Base converter class for learned rounding quantization.

Provides shared infrastructure for LearnedRoundingConverter (FP8/INT8)
and LearnedNVFP4Converter (NVFP4). Contains common initialization,
SVD computation, LR scheduling, and early stopping logic.
"""
import gc
import math
import re
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict

import torch

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE
from ..pinned_transfer import transfer_to_gpu_pinned


class BaseLearnedConverter(ABC):
    """
    Abstract base class for learned rounding quantization converters.

    Provides shared infrastructure:
    - SVD computation for principal component optimization
    - Tier-based adaptive LR scheduling
    - Early stopping with configurable thresholds

    Subclasses must implement:
    - convert(): Format-specific quantization logic
    """

    def __init__(
        self,
        optimizer: str = "original",
        num_iter: int = 1000,
        lr: float = 8.077300000003e-3,
        top_p: float = 0.01,
        min_k: int = 1,
        max_k: int = 16,
        full_matrix: bool = False,
        no_learned_rounding: bool = False,
        lr_schedule: str = "adaptive",
        lr_gamma: float = 0.99,
        lr_patience: int = 50,
        lr_factor: float = 0.95,
        lr_min: float = 1e-8,
        lr_cooldown: int = 0,
        lr_threshold: float = 0.0,
        lr_adaptive_mode: str = "simple-reset",
        lr_shape_influence: float = 1.0,
        lr_threshold_mode: str = "rel",
        lr_small_mult: Optional[float] = None,
        early_stop_loss: float = 1e-8,
        early_stop_lr: float = 1e-10,
        early_stop_stall: int = 1000,
        device: Optional[str] = None,
        # LoRA extraction parameters
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: Optional[str] = None,
        lora_ar_threshold: float = 0.0,
        **kwargs,
    ):
        """
        Initialize base converter with shared optimization parameters.

        Args:
            optimizer: Optimization algorithm ("original", "adamw", "radam")
            num_iter: Number of optimization iterations
            top_p: Proportion of principal components to use
            min_k: Minimum number of SVD components
            max_k: Maximum number of SVD components
            full_matrix: Use full SVD instead of lowrank
            no_learned_rounding: Skip optimization, use simple quantization
            lr_schedule: LR schedule ("adaptive", "exponential", "plateau")
            lr_gamma: Decay factor for exponential schedule
            lr_patience: Steps before decay for plateau schedule
            lr_factor: LR reduction factor for plateau
            lr_min: Minimum learning rate
            lr_cooldown: Cooldown steps after LR reduction
            lr_threshold: Minimum improvement threshold
            lr_adaptive_mode: Counter reset mode ("simple-reset", "no-reset")
            lr_shape_influence: Shape-aware LR scaling (0.0-1.0)
            lr_threshold_mode: Threshold mode ("rel", "abs")
            lr_small_mult: Dimension-aware multiplier (default 1.0)
            early_stop_loss: Stop when loss drops below this
            early_stop_lr: Stop when LR drops below this
            early_stop_stall: Stop after this many steps without improvement
            device: Device to use for optimization (default: auto-detect)
            **kwargs: Additional optimizer-specific parameters (e.g., lr)
        """
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SVD parameters
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.full_matrix = full_matrix

        # Optimizer configuration
        self.optimizer_choice = optimizer
        self.num_iter = num_iter
        self.lr = lr
        self.no_learned_rounding = no_learned_rounding
        self.optimizer_kwargs = kwargs

        # LR schedule configuration
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_cooldown = lr_cooldown
        self.lr_threshold = lr_threshold
        self.lr_adaptive_mode = lr_adaptive_mode
        self.lr_shape_influence = lr_shape_influence
        self.lr_threshold_mode = lr_threshold_mode
        self.lr_small_mult = lr_small_mult

        # Early stopping thresholds
        self.early_stop_loss = early_stop_loss
        self.early_stop_lr = early_stop_lr
        self.early_stop_stall = early_stop_stall

        # LoRA extraction configuration
        self.extract_lora = extract_lora
        self.lora_rank = lora_rank
        self.lora_depth = lora_depth
        self.lora_ar_threshold = lora_ar_threshold
        self.lora_target_regex = None
        if lora_target:
            import re
            try:
                self.lora_target_regex = re.compile(lora_target)
            except re.error:
                from ..utils.logging import warning
                warning(f"      [LoRA] Invalid target regex '{lora_target}', ignoring.")

    def _should_extract_lora(self, key: str, shape: torch.Size, depth: int = -1) -> bool:
        """
        Determine if LoRA should be extracted for the given layer.

        Heuristics:
        1. Explicitly enabled via self.extract_lora.
        2. Key matches lora_target_regex (if provided).
        3. Block index < lora_depth (e.g. .0. blocks).
        4. Sensitive layers (qkv, proj, attn) that are not too skewed.
        """
        if not self.extract_lora:
            return False

        # 1. Explicit Regex Match
        if self.lora_target_regex and self.lora_target_regex.search(key):
            return True

        # 2. Block Index Check
        block_idx = depth
        if block_idx == -1:
            # Try to extract from key if not provided
            import re
            block_match = re.search(r'\.(\d+)\.', key)
            if block_match:
                block_idx = int(block_match.group(1))

        if block_idx != -1:
            # Global Depth Limit: depth=1 targets only block 0
            if block_idx >= self.lora_depth:
                return False

            # Calculate Aspect Ratio
            if len(shape) >= 2:
                rows, cols = shape[0], shape[1]
                ar = max(rows, cols) / min(rows, cols)
                
                # 3. User Aspect Ratio Threshold Override (LESS THAN targets square layers)
                if self.lora_ar_threshold > 0.0:
                    return ar < self.lora_ar_threshold

                # Case 3: AR > 4.0 -> Too large. Never extract (default logic).
                if ar > 4.0:
                    return False
                    
                # Case 1: AR < 3.0 -> Safe zone. Extract if below depth (default logic).
                if ar < 3.0:
                    return True
                    
                # Case 2: 3.0 <= AR <= 4.0 -> Marginal zone.
                # Extract if Block 0 OR specific sensitive type (QKV/Attn).
                if block_idx == 0:
                    return True
                
                k_lower = key.lower()
                if "qkv" in k_lower or "attn" in k_lower or "proj" in k_lower:
                    return True
                    
                return False

        return False

    def _extract_error_lora(self, W_orig: torch.Tensor, W_dequant: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract quantization error into low-rank components (U, V).
        Error = W_orig - W_dequant
        """
        if self.lora_rank <= 0:
            return None

        with torch.no_grad():
            # Ensure everything is on same device and float32 for SVD
            error = (W_orig - W_dequant).to(device=W_orig.device, dtype=torch.float32)
            
            # Flatten if necessary (for convs)
            if error.ndim > 2:
                error = error.flatten(1)
                
            M, N = error.shape
            actual_rank = min(self.lora_rank, M, N)

            if actual_rank <= 0:
                return None

            try:
                # Use svd_lowrank for efficiency
                U, S, V = torch.svd_lowrank(error, q=actual_rank, niter=4)
                
                # LoRA Up = U * diag(S)
                # LoRA Down = V^T
                # Return as float16 CPU tensors for storage efficiency
                return {
                    "lora_up": (U @ torch.diag(S)).to(torch.float32).cpu().contiguous(),
                    "lora_down": V.t().to(torch.float32).cpu().contiguous(),
                }
            except Exception as e:
                from ..utils.logging import warning
                warning(f"      [LoRA] SVD failed for error extraction: {e}")
                return None

    def _compute_svd_components(
        self, W_float32: torch.Tensor, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute SVD components for optimization.

        Args:
            W_float32: Input tensor (2D)
            verbose: Print SVD computation info

        Returns:
            Tuple of (U_k, Vh_k, k) where k is number of components used
        """
        M, N = W_float32.shape
        max_rank = min(M, N)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        if verbose:
            print(f"    - Tensor shape: [{M}, {N}], Max rank: {max_rank}. Using k={k} components.")

        if self.full_matrix:
            if verbose:
                print("    - Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                if verbose:
                    print("    - Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                if verbose:
                    print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        return U[:, :k], Vh[:k, :], k

    def _adaptive_lr_update_cosine(
        self,
        curr_lr: float,
        improved: bool,
        worse_loss_counter: int,
        iteration: int,
        tensor_shape: Tuple[int, int],
        min_lr: float = 1e-10,
        small_mult: float = 1.0,
    ) -> Tuple[float, bool]:
        """
        Cosine-based adaptive LR update with shape-awareness.

        Uses a U-shaped cosine curve for both boost and decay that is:
        - Gentle at start of stall (multiplier close to 1.0)
        - Most aggressive at midpoint of early_stop_stall
        - Gentle again near end to prevent bottoming out too early

        Decay is only applied every `lr_cooldown` steps to prevent
        compounding too rapidly over thousands of iterations.

        IMPLEMENTATION LOCATIONS:
        This method should be called from the adaptive LR branch in:
        - learned_rounding.py: _optimize_original() ~line 417
        - learned_rounding.py: _optimize_int8_original() ~line 1020
        - learned_mxfp8.py: _optimize_original() ~line 430
        - learned_nvfp4.py: _optimize_original() ~line 490

        Replace the existing tier-based if/elif chain with:
            new_lr, lr_updated = self._adaptive_lr_update_cosine(
                curr_lr, improved, worse_loss_counter, iteration,
                (M, N), self.early_stop_lr, small_mult
            )
            if lr_updated:
                curr_lr = new_lr

        Args:
            curr_lr: Current learning rate
            improved: Whether loss improved this iteration
            worse_loss_counter: Steps since last improvement (BEFORE reset if improved)
            iteration: Current optimization iteration
            tensor_shape: (M, N) dimensions of weight tensor
            min_lr: Minimum allowed learning rate
            small_mult: Dimension-aware multiplier (default 1.0)

        Returns:
            Tuple of (new_lr, lr_was_updated)
        """
        M, N = tensor_shape
        shape_ratio = abs(M - N) / max(M, N)  # 0 for square, ~1 for very skewed

        # Cosine U-curve: gentle at start/end of stall, aggressive at midpoint
        t = min(worse_loss_counter / max(self.early_stop_stall, 1), 1.0)
        u_factor = (1 + math.cos(2 * math.pi * t)) / 2

        if improved:
            # Boost: scale distance from 1.0 by shape factor
            # More skewed tensors get slightly stronger boost
            base_boost = 1.25
            distance = base_boost - 1.0
            scaled_distance = distance * (1.0 + 0.5 * shape_ratio)

            # Aggressiveness: gentle at start of stall (u_factor near 1.0)
            # Most aggressive at midpoint (u_factor near 0.0)
            # This prevents compounding boosts when already improving rapidly.
            boost_mult = 1.0 + scaled_distance * (1.0 - u_factor)

            # Apply small_mult
            boost_mult *= small_mult

            new_lr = min(curr_lr * boost_mult, 100.0)
            return new_lr, True
        else:
            # Decay: only update every lr_cooldown steps
            cooldown = max(self.lr_cooldown, 1)
            if iteration % cooldown != 0:
                return curr_lr, False

            # Decay range: shape-aware
            # More skewed tensors = slightly stronger decay at midpoint
            # Uses self.lr_factor as base (default 0.95)
            min_decay = self.lr_factor - 0.03 * shape_ratio
            max_decay = 0.995

            decay_mult = min_decay + (max_decay - min_decay) * u_factor
            
            # Apply small_mult
            decay_mult *= small_mult
            
            new_lr = max(curr_lr * decay_mult, min_lr)
            return new_lr, True

    def _compute_shape_aware_plateau_params(
        self, M: int, N: int
    ) -> Tuple[int, float, int]:
        """
        Compute shape-aware parameters for plateau LR schedule.

        Args:
            M: First dimension
            N: Second dimension

        Returns:
            Tuple of (effective_patience, effective_factor, effective_cooldown)
        """
        if self.lr_schedule == "plateau" and self.lr_shape_influence > 0:
            aspect_ratio = max(M, N) / min(M, N)
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence

            effective_patience = self.lr_patience
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor ** ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        return effective_patience, effective_factor, effective_cooldown

    def _check_improvement(
        self, current_loss: float, best_loss: float
    ) -> bool:
        """
        Check if current loss is a significant improvement.

        Supports both relative and absolute threshold modes.

        Args:
            current_loss: Current iteration loss
            best_loss: Best loss seen so far

        Returns:
            True if improvement is significant
        """
        if self.lr_threshold > 0:
            if self.lr_threshold_mode == "rel":
                return current_loss < best_loss * (1.0 - self.lr_threshold)
            else:  # 'abs'
                return (best_loss - current_loss) > self.lr_threshold
        return current_loss < best_loss

    def _cleanup_tensors(self, *tensors) -> None:
        """Delete tensors and clear GPU cache."""
        for t in tensors:
            del t
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @abstractmethod
    def convert(self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1) -> Tuple:
        """
        Convert tensor to quantized format.

        Args:
            W_orig: Input tensor (2D)
            key: Layer name for filtering/heuristics
            depth: Block depth for filtering

        Returns:
            Tuple of quantized tensors (format-specific) + optional extra_tensors dict
        """
        pass
