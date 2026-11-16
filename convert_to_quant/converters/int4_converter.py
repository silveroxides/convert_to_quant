"""
INT4 ConvRot W4A4 converter for convert_to_quant.

Implements row-wise symmetric INT4 quantization with group-wise Hadamard rotation (ConvRot),
SVD-guided AdaRound (learned soft rounding), and 2-per-byte row-major packing for ComfyUI's
convrot_w4a4 runtime format.
"""

import gc
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import (
    COMPUTE_DTYPE,
    INT4_SYMMETRIC_MAX,
    SCALE_DTYPE,
    TARGET_INT4_DTYPE,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.convrot import (
    build_hadamard,
    find_max_compatible_group_size,
    pack_int4_row_major,
    quantize_int4_rowwise,
    rotate_weight,
)
from ..utils.logging import debug, info, verbose
from ..utils.tensor_utils import prepare_calibration_data
from .base_converter import BaseLearnedConverter


class INT4Converter(BaseLearnedConverter):
    """
    Converter for row-wise INT4 ConvRot (W4A4) quantization.

    Supports:
    - Group-wise Hadamard rotation (static or dynamic group size).
    - Fast simple RTN quantization (--simple).
    - SVD-guided AdaRound (learned soft rounding) with Prodigy, AdamW, RAdam, or SGD.
    - Optional DualRound 2-pass scale refinement (--scale_optimization dualround).
    - Residual bias calibration.
    - 2-per-byte row-major packing.
    """

    def __init__(
        self,
        convrot: bool = True,
        convrot_group_size: int = 64,
        dynamic_convrot: bool = True,
        device: Optional[str] = None,
        no_learned_rounding: bool = True,
        calib_scale: float = 1.0,
        has_bias: bool = False,
        scale_optimization: str = "fixed",
        optimizer: str = "prodigy",
        num_iter: int = 2000,
        lr: float = 1.0,
        top_p: float = 0.2,
        min_k: int = 128,
        max_k: int = 1280,
        full_matrix: bool = False,
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
        use_speed: bool = False,
        **kwargs,
    ):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            optimizer=optimizer,
            num_iter=num_iter,
            lr=lr,
            top_p=top_p,
            min_k=min_k,
            max_k=max_k,
            full_matrix=full_matrix,
            no_learned_rounding=no_learned_rounding,
            lr_schedule=lr_schedule,
            lr_gamma=lr_gamma,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_min=lr_min,
            lr_cooldown=lr_cooldown,
            lr_threshold=lr_threshold,
            lr_adaptive_mode=lr_adaptive_mode,
            lr_shape_influence=lr_shape_influence,
            lr_threshold_mode=lr_threshold_mode,
            lr_small_mult=lr_small_mult,
            early_stop_loss=early_stop_loss,
            early_stop_lr=early_stop_lr,
            early_stop_stall=early_stop_stall,
            device=resolved_device,
            use_speed=use_speed,
            **kwargs,
        )
        self.convrot = convrot
        self.convrot_group_size = convrot_group_size
        self.dynamic_convrot = dynamic_convrot
        self.calib_scale = calib_scale
        self.has_bias = has_bias
        self.scale_optimization = scale_optimization
        self.optimizer_choice = optimizer
        self.block_size = None
        self.last_group_size: Optional[int] = None
        self._current_extra_tensors: Dict[str, torch.Tensor] = {}

    def _optimize_int4_adaround(
        self,
        W_rot: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        X_rot: torch.Tensor,
        Y_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SVD-guided AdaRound (learned soft rounding) optimization over the rotated parameter space
        minimizing local activation reconstruction MSE using the cached calibration data.
        """
        M, N = W_rot.shape

        # 1. Compute SVD components of rotated parameter matrix
        U_k, Vh_k, k = self._compute_svd_components(W_rot, verbose=True)

        # 2. Setup soft rounding parameters V
        scale_broadcast = scale.unsqueeze(1) if scale.dim() == 1 else scale
        W_scaled = W_rot / scale_broadcast.clamp_min(1e-12)
        W_floor = W_scaled.floor()

        # Target fraction for soft rounding
        target = W_scaled - W_floor
        target = torch.clamp(target, min=1e-6, max=1.0 - 1e-6)

        # Temperature schedule for sigmoid (start soft, sharpen over time)
        T_start, T_end = 20.0, 2.0
        V_init = -torch.log((1.0 / target) - 1.0) * T_start
        V = V_init.clone().detach().requires_grad_(True)

        # 3. Setup optimizer
        curr_lr = self.lr
        if self.optimizer_choice == "adamw":
            optimizer = AdamW([V], lr=curr_lr)
        elif self.optimizer_choice == "radam":
            optimizer = RAdam([V], lr=curr_lr)
        elif self.optimizer_choice == "prodigy":
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

            optimizer = ProdigyPlusScheduleFree([V], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed)
        else:
            optimizer = None  # Manual SGD

        # 4. Compute initial metrics for dynamic self-tuning regularization
        with torch.no_grad():
            init_W_q_rounded = qdata.to(COMPUTE_DTYPE)
            init_W_rounded_dequant = init_W_q_rounded * scale_broadcast
            init_mse_rounded = torch.nn.functional.mse_loss(X_rot @ init_W_rounded_dequant.T, Y_ref)
            init_svd_rounded = torch.linalg.norm(U_k.T @ (init_W_rounded_dequant - W_rot) @ Vh_k.T)

        # SVD regularization balance factor: ~1% of initial rounded MSE loss
        if init_svd_rounded.item() > 1e-8:
            alpha_svd = 0.01 * (init_mse_rounded.item() / init_svd_rounded.item())
        else:
            alpha_svd = 0.0

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_V = V.detach().clone()
        best_converged_ratio = 0.0
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        effective_patience, effective_factor, effective_cooldown = None, None, None
        if schedule_name == "plateau":
            effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(M, N)

        decay_factor = self.lr_factor if self.lr_factor is not None else 0.95
        if decay_factor >= 1.0:
            decay_factor = 0.95

        window_size = max(5, int(2.5 / (1.0 - decay_factor)))
        loss_span_threshold = self.early_stop_loss / (1.0 - decay_factor)
        target_converged_ratio = 0.90

        loss_history = []
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (AdaRound-INT4-{self.optimizer_choice}-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            if optimizer is not None:
                optimizer.zero_grad()

            # Forward pass: Optimized soft rounding (smooth AdaRound)
            temp = T_start + (T_end - T_start) * (i / self.num_iter)
            h_V = torch.sigmoid(V / temp)
            W_q = W_floor + h_V
            W_dequant = W_q * scale_broadcast

            # Discretization and Convergence early stopping check
            converged_ratio = ((torch.sigmoid(V) < 0.05) | (torch.sigmoid(V) > 0.95)).float().mean().item()

            # Loss 1: Output activation MSE
            Y_pred = X_rot @ W_dequant.T
            loss_mse = torch.nn.functional.mse_loss(Y_pred, Y_ref)

            # Loss 2: SVD-guided weight-space projection error
            weight_error = W_dequant - W_rot
            projected_error = U_k.T @ weight_error @ Vh_k.T
            loss_svd = torch.linalg.norm(projected_error)

            # Loss 3: Soft rounding binary regularizer
            loss_reg = (1.0 - (2.0 * h_V - 1.0).pow(2)).mean()

            # Total Loss - Normalized
            loss_mse_scaled = loss_mse / max(init_mse_rounded.item(), 1e-12)
            if alpha_svd > 0 and init_svd_rounded.item() > 1e-8:
                loss_svd_scaled = loss_svd / init_svd_rounded.item()
            else:
                loss_svd_scaled = 0.0

            loss = loss_mse_scaled + 0.01 * loss_svd_scaled + 0.1 * loss_reg

            # Scale up loss for backpropagation to prevent float32 underflow on large layers
            scaled_loss = loss * 1e5

            if optimizer is not None:
                scaled_loss.backward()
                if V.grad is not None:
                    V.grad.div_(1e5)
                optimizer.step()
            else:
                if V.grad is not None:
                    V.grad.zero_()
                scaled_loss.backward()
                with torch.no_grad():
                    V -= curr_lr * (V.grad / 1e5)

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            loss_history.append(current_loss_val)
            if len(loss_history) > window_size:
                loss_history.pop(0)

            # Saturated Flatline Trigger
            if converged_ratio >= target_converged_ratio and len(loss_history) == window_size:
                loss_span = max(loss_history) - min(loss_history)
                if loss_span < loss_span_threshold:
                    verbose(
                        f"\n      - Discretization early stop: {converged_ratio*100:.2f}% parameters converged. Loss span: {loss_span:.2e} (< {loss_span_threshold:.2e}). Stopping."
                    )
                    break

            if improved:
                best_loss = current_loss_val
                best_V = V.detach().clone()
                best_converged_ratio = converged_ratio
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Schedule-based learning rate adjustments
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        if optimizer is not None:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = curr_lr
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")
            else:  # "adaptive"
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Postfix
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}",
                    }
                )

            # Early stopping conditions
            if (
                best_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        # Discretize V strictly to INT4 symmetric range [-7, 7]
        with torch.no_grad():
            final_h_V = torch.sigmoid(best_V)
            final_round_up = (final_h_V >= 0.5).to(COMPUTE_DTYPE)
            final_qdata = (
                (W_floor + final_round_up)
                .clamp(-INT4_SYMMETRIC_MAX, INT4_SYMMETRIC_MAX)
                .round()
                .to(TARGET_INT4_DTYPE)
            )
            verbose(
                f"    - Discretization audit: {best_converged_ratio * 100:.2f}% of parameters converged to strict boundaries."
            )

        self._cleanup_tensors(U_k, Vh_k, V, best_V, W_scaled, W_floor)
        del U_k, Vh_k, V, best_V, W_scaled, W_floor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_qdata, scale

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        calibration_data: Optional[torch.Tensor] = None,
        has_bias: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert a 2D weight tensor to packed INT4 ConvRot W4A4.

        Args:
            W_orig: Original 2D weight tensor.
            key: Optional layer name for diagnostics.
            depth: Optional layer depth.
            calibration_data: Optional simulated/real calibration activation tensor.
            has_bias: Optional override for whether layer has a bias parameter.

        Returns:
            Tuple of (packed_qdata, scale, dequantized_weight, extra_tensors):
            - packed_qdata: torch.int8 of shape (rows, cols // 2)
            - scale: torch.float32 of shape (rows,)
            - dequantized_weight: torch.float32 of shape (rows, cols) in original (un-rotated) basis
            - extra_tensors: Dict containing bias_correction if computed
        """
        layer_has_bias = self.has_bias if has_bias is None else has_bias
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            verbose("  - Tensor is all zeros, skipping optimization.")
            rows, cols = W_float32.shape
            packed_zeros = torch.zeros((rows, cols // 2), dtype=TARGET_INT4_DTYPE, device=self.device)
            scales = torch.ones(rows, dtype=SCALE_DTYPE, device=self.device)
            dequant_zeros = torch.zeros_like(W_float32)
            self.last_group_size = self.convrot_group_size
            return packed_zeros, scales, dequant_zeros, {}

        rows, N = W_float32.shape
        if N % 2 != 0:
            raise ValueError(f"INT4 ConvRot requires even in_features, got {N}")

        # Resolve group size
        layer_group_size = self.convrot_group_size
        if self.dynamic_convrot:
            min_grp = min(self.convrot_group_size, 64)
            resolved = find_max_compatible_group_size(N, min_group_size=min_grp)
            if resolved is not None:
                layer_group_size = resolved
        elif N % layer_group_size != 0 and N % 64 == 0:
            layer_group_size = 64

        if layer_group_size is None or N % layer_group_size != 0:
            raise ValueError(
                f"INT4 ConvRot W4A4 requires in_features ({N}) divisible by group_size ({layer_group_size})."
            )

        self.last_group_size = layer_group_size

        # Apply group-wise Hadamard rotation
        H = build_hadamard(layer_group_size, device=self.device, dtype=COMPUTE_DTYPE)
        W_rot = rotate_weight(W_float32, H, layer_group_size)
        verbose(f"    - Applied ConvRot Hadamard rotation (group_size={layer_group_size}).")

        # Prepare calibration activations if provided
        X_rot = None
        Y_ref = None
        if calibration_data is not None:
            X_rot, Y_ref, _ = prepare_calibration_data(
                W_rot,
                calibration_data,
                True,
                layer_group_size,
                self.device,
                COMPUTE_DTYPE,
                calib_scale=self.calib_scale,
            )

        # Baseline quantization (simple RTN)
        qdata, scale = quantize_int4_rowwise(W_rot)

        # Learned AdaRound optimization path
        if not self.no_learned_rounding and self.num_iter > 0 and X_rot is not None:
            if self.scale_optimization == "dualround":
                verbose("    - Scale Optimization: DUALROUND (Pass 1)")
                qdata, scale = self._optimize_int4_adaround(W_rot, qdata, scale, X_rot, Y_ref)

                verbose("    - Scale Optimization: Re-estimating scales based on Pass 1 output...")
                scale_broadcast = scale.unsqueeze(1) if scale.dim() == 1 else scale
                dequant_opt = qdata.to(COMPUTE_DTYPE) * scale_broadcast
                row_max_opt = dequant_opt.abs().amax(dim=-1)
                scale_opt = (row_max_opt.clamp_min(1e-12) / float(INT4_SYMMETRIC_MAX)).to(SCALE_DTYPE)
                qdata, _ = quantize_int4_rowwise(W_rot)
                scale = scale_opt

                del dequant_opt, row_max_opt, scale_opt
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                verbose("    - Scale Optimization: DUALROUND (Pass 2)")
                qdata, scale = self._optimize_int4_adaround(W_rot, qdata, scale, X_rot, Y_ref)
            else:
                verbose("    - Applying SVD-guided AdaRound optimization for INT4 ConvRot W4A4...")
                qdata, scale = self._optimize_int4_adaround(W_rot, qdata, scale, X_rot, Y_ref)

        # Dequantize in rotated space: dq_rot = qdata * scale
        scale_broadcast = scale.unsqueeze(1) if scale.dim() == 1 else scale
        dequantized_rotated = qdata.to(COMPUTE_DTYPE) * scale_broadcast

        # Un-rotate to return dequantized weights in the original basis
        dequantized = rotate_weight(dequantized_rotated, H, layer_group_size)

        extra_tensors: Dict[str, torch.Tensor] = {}

        # Residual bias calibration
        if layer_has_bias and X_rot is not None and Y_ref is not None:
            with torch.no_grad():
                Y_quant = X_rot @ dequantized_rotated.T
                bias_adj = (Y_ref - Y_quant).mean(dim=0)
                extra_tensors["bias_correction"] = bias_adj.cpu()
                verbose(
                    f"    - Phase 4: Residual Bias Calibration (bias correction norm: {bias_adj.norm().item():.6f})"
                )

        # Pack signed int4 (in int8 container) 2-per-byte row-major
        packed = pack_int4_row_major(qdata)

        self._cleanup_tensors(W_float32, W_rot, X_rot, Y_ref, dequantized_rotated, H)

        return (
            packed,
            scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized,
            extra_tensors,
        )
