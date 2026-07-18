"""
Base converter class for learned rounding quantization.

Provides shared infrastructure for LearnedRoundingConverter (FP8/INT8)
and LearnedNVFP4Converter (NVFP4). Contains common initialization,
SVD computation, LR scheduling, and early stopping logic.
"""

import gc
import math
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Dict,
    Optional,
    Tuple,
)

import torch

from .convergence import (
    AdaptiveConvergenceController,
    TuningReportCollector,
    convergence_window,
)


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
        optimizer: str = "prodigy",
        num_iter: int = 2000,
        lr: float = 1.0,
        top_p: float = 0.2,
        min_k: int = 128,
        max_k: int = 1280,
        full_matrix: bool = False,
        no_learned_rounding: bool = False,
        lr_schedule: str = "plateau",
        lr_gamma: float = 0.99,
        lr_patience: int = 1,
        lr_factor: float = 0.95,
        lr_min: float = 1e-8,
        lr_cooldown: int = 0,
        lr_threshold: float = 0.0,
        lr_adaptive_mode: str = "simple-reset",
        lr_shape_influence: float = 1.0,
        lr_threshold_mode: str = "rel",
        lr_small_mult: Optional[float] = None,
        early_stop_loss: float = 5e-9,
        early_stop_lr: float = 1.01e-8,
        early_stop_stall: int = 2000,
        auto_tune: bool = False,
        auto_tune_report: Optional[str] = None,
        tuning_report_collector: Optional[TuningReportCollector] = None,
        device: Optional[str] = None,
        # LoRA extraction parameters
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: Optional[str] = None,
        lora_ar_threshold: float = 0.0,
        use_speed: bool = False,
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
            auto_tune: Probe and adapt convergence within num_iter
            auto_tune_report: Optional JSON diagnostics output path
            tuning_report_collector: Shared collector for multi-format conversion
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
        self.use_speed = use_speed
        if self.optimizer_choice == "prodigy":
            try:
                import prodigyplus.prodigy_plus_schedulefree
            except ImportError:
                raise ImportError("User needs to run `pip install prodigy-plus-schedule-free` to use the prodigy optimizer.")

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

        # Automatic convergence tuning is opt-in. Manual mode retains every
        # scheduler and early-stop value above without passing through the
        # automatic controller.
        self.auto_tune = auto_tune
        self.auto_tune_report = auto_tune_report
        self._tuning_report_collector = tuning_report_collector or TuningReportCollector(auto_tune_report)
        self._active_auto_controller: Optional[AdaptiveConvergenceController] = None
        self._active_layer_key: Optional[str] = None

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

    def _run_selected_optimizer(self, *args, method_prefix: str = "_optimize_", **kwargs):
        """Dispatch to the configured learned optimizer implementation."""
        if self.optimizer_choice not in {"original", "adamw", "radam", "prodigy"}:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")
        method_name = f"{method_prefix}{self.optimizer_choice}"
        method = getattr(self, method_name)
        if not self.auto_tune or self.num_iter <= 0:
            return method(*args, **kwargs)
        return self._run_auto_tuned_optimizer(method, args, kwargs, method_name)

    def _run_auto_tuned_optimizer(self, method, args, kwargs, method_name: str):
        """Run LR probes, one selected attempt, and at most one recovery."""
        weight = next((arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim >= 2), None)
        if weight is None:
            return method(*args, **kwargs)

        shape = (int(weight.shape[0]), int(weight.shape[1]))
        rank = None
        matrix_args = [arg for arg in args[1:] if isinstance(arg, torch.Tensor) and arg.ndim == 2]
        for left_index, left in enumerate(matrix_args):
            if tuple(left.shape) == tuple(weight.shape) or left.shape[0] != shape[0]:
                continue
            for right in matrix_args[left_index + 1 :]:
                if left.shape[1] == right.shape[0] and right.shape[1] == shape[1]:
                    rank = int(left.shape[1])
                    break
            if rank is not None:
                break
        rank = rank or min(shape)

        original = {
            "num_iter": self.num_iter,
            "lr": self.lr,
            "lr_schedule": self.lr_schedule,
            "lr_adaptive_mode": self.lr_adaptive_mode,
            "early_stop_loss": self.early_stop_loss,
            "early_stop_lr": self.early_stop_lr,
            "early_stop_stall": self.early_stop_stall,
        }
        total_budget = self.num_iter
        anchor_lr = self.lr
        window = convergence_window(shape, rank)
        probe_steps = max(8, window // 2)
        use_probes = total_budget >= 6 * probe_steps
        attempts = []
        selected_lr = anchor_lr
        consumed = 0
        result = None
        result_loss = math.inf
        retried = False

        cpu_rng = torch.random.get_rng_state()
        cuda_device = weight.device if weight.is_cuda else None
        cuda_rng = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None

        def restore_rng() -> None:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None and cuda_device is not None:
                torch.cuda.set_rng_state(cuda_rng, cuda_device)

        def run_attempt(lr: float, budget: int, kind: str):
            restore_rng()
            self.num_iter = budget
            self.lr = lr
            self.lr_schedule = "adaptive"
            self.lr_adaptive_mode = "simple-reset"
            self.early_stop_loss = -1.0
            self.early_stop_lr = 1e-12
            self.early_stop_stall = budget + 1
            controller = AdaptiveConvergenceController(
                shape=shape,
                rank=rank,
                optimizer=self.optimizer_choice,
                initial_lr=lr,
                budget=budget,
                kind=kind,
            )
            self._active_auto_controller = controller
            try:
                attempt_result = method(*args, **kwargs)
            finally:
                self._active_auto_controller = None
            attempts.append(controller.summary)
            return attempt_result, controller

        try:
            probe_results = []
            if use_probes:
                for multiplier in (0.25, 1.0, 4.0):
                    candidate_lr = anchor_lr * multiplier
                    _, controller = run_attempt(candidate_lr, probe_steps, "probe")
                    probe_results.append((controller.score(), candidate_lr, controller))
                    consumed += controller.summary.iterations
                stable = [item for item in probe_results if not item[2].retry_recommended]
                candidates = stable or probe_results
                selected_lr = max(candidates, key=lambda item: item[0])[1]

            remaining = max(1, total_budget - consumed)
            result, controller = run_attempt(selected_lr, remaining, "selected")
            consumed += controller.summary.iterations
            result_loss = controller.best_loss

            remaining = total_budget - consumed
            if controller.retry_recommended and remaining >= max(8, window // 2):
                retried = True
                stable_probe_lrs = [
                    summary.lr for summary in attempts
                    if summary.kind == "probe" and summary.retry_reason is None
                ]
                retry_lr = min(stable_probe_lrs) if stable_probe_lrs else anchor_lr * 0.1
                retry_result, retry_controller = run_attempt(retry_lr, remaining, "retry")
                consumed += retry_controller.summary.iterations
                if retry_controller.best_loss < result_loss:
                    result = retry_result
                    result_loss = retry_controller.best_loss

            selected_summary = min(
                (summary for summary in attempts if summary.kind in {"selected", "retry"}),
                key=lambda summary: summary.best_loss if summary.best_loss is not None else math.inf,
            )
            record = {
                "layer": self._active_layer_key or "<unknown>",
                "shape": list(shape),
                "rank": rank,
                "converter": type(self).__name__,
                "optimizer": self.optimizer_choice,
                "method": method_name,
                "budget": total_budget,
                "iterations": consumed,
                "window": window,
                "lr_anchor": anchor_lr,
                "selected_lr": selected_lr,
                "retried": retried,
                "stop_reason": selected_summary.stop_reason,
                "best_loss": selected_summary.best_loss,
                "normalized_best_loss": selected_summary.normalized_best_loss,
                "attempts": [summary.as_dict() for summary in attempts],
            }
            self._tuning_report_collector.add(record)
            from ..utils.logging import info

            retry_note = ", retry used" if retried else ""
            info(
                f"    - Auto tune: lr={selected_lr:.3g}, iterations={consumed}/{total_budget}, "
                f"stop={selected_summary.stop_reason}{retry_note}"
            )
            return result
        finally:
            self._active_auto_controller = None
            for name, value in original.items():
                setattr(self, name, value)

    def get_tuning_report(self) -> Dict[str, object]:
        """Return structured diagnostics collected by automatic tuning."""
        return self._tuning_report_collector.as_dict()

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

            block_match = re.search(r"\.(\d+)\.", key)
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
                    "lora_down": V.t().to(torch.float32).cpu().contiguous()
                }
            except Exception as e:
                from ..utils.logging import warning

                warning(f"      [LoRA] SVD failed for error extraction: {e}")
                return None

    def _compute_svd_components(self, W_float32: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
        self, curr_lr: float, improved: bool, worse_loss_counter: int, iteration: int, tensor_shape: Tuple[int, int],
        min_lr: float = 1e-10, small_mult: float = 1.0
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
        if self._active_auto_controller is not None:
            if self._active_auto_controller.should_stop:
                return 0.0, True
            return self._active_auto_controller.update_lr(curr_lr)

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

    def _compute_shape_aware_plateau_params(self, M: int, N: int) -> Tuple[int, float, int]:
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
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        return effective_patience, effective_factor, effective_cooldown

    def _check_improvement(self, current_loss: float, best_loss: float) -> bool:
        """
        Check if current loss is a significant improvement.

        Supports both relative and absolute threshold modes.

        Args:
            current_loss: Current iteration loss
            best_loss: Best loss seen so far

        Returns:
            True if improvement is significant
        """
        if self._active_auto_controller is not None:
            improved = self._active_auto_controller.observe(current_loss, best_loss)
            if self._active_auto_controller.should_stop:
                # Every learned loop already checks these thresholds, including
                # optimizer warmup branches that intentionally skip LR updates.
                self.early_stop_loss = math.inf
                self.early_stop_lr = math.inf
            return improved

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
    def convert(self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1, **kwargs) -> Tuple:
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
