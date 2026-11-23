"""
W6A8 ConvRot converter for convert_to_quant.

Implements uniform symmetric 6-bit grouped quantization with group-wise Hadamard
rotation (ConvRot), dual scaling (fp8/fp32 per-group s_rel + fp32 per-channel s_channel),
SVD-guided AdaRound (learned soft rounding), and 6-bit (nibble + 2-bit high plane)
packing for ComfyUI's w6a8_int8 (AsymW4A8Int8Layout) runtime format.
"""

import gc
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE, TARGET_W6A8_DTYPE, W6A8_DEFAULT_CONVROT_GROUP_SIZE, W6A8_DEFAULT_GROUP_SIZE, W6A8_LEVELS, W6A8_SCALE_REL_DTYPE
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.convrot import build_hadamard, dequantize_w6a8_rotated, pack_int6_w6a8, quantize_w6a8_grouped, resolve_w6a8_convrot_group_size, rotate_weight, validate_w6a8_shape
from ..utils.logging import debug, info, verbose
from ..utils.tensor_utils import prepare_calibration_data
from .base_converter import BaseLearnedConverter


class W6A8Converter(BaseLearnedConverter):
    """
    Converter for W6A8 (6-bit uniform weight, INT8 activation) ConvRot quantization.

    Targets ComfyUI's ``w6a8_int8`` format (``AsymW4A8Int8Layout`` with ``bits=6``):
    - Group-wise Hadamard rotation (default 256, fallback to compatible power of 4).
    - Uniform symmetric 6-bit levels [-31, 31] stored as q + 32 in [1, 63].
    - Per-group relative scale ``s_rel`` (default float8_e4m3fn) and per-channel
      ``s_channel`` (float32).
    - Fast simple quantization (--simple) matching comfy-kitchen's 2-pass ALS +
      grid-aware group-scale search bit-for-bit.
    - SVD-guided AdaRound (learned soft rounding) with Prodigy, AdamW, RAdam, or SGD,
      plus optional DualRound scale refinement.
    """

    def __init__(
        self,
        convrot: bool = True,
        convrot_group_size: int = W6A8_DEFAULT_CONVROT_GROUP_SIZE,
        group_size: int = W6A8_DEFAULT_GROUP_SIZE,
        block_size: Optional[int] = None,
        dynamic_convrot: bool = True,
        scale_dtype: torch.dtype = W6A8_SCALE_REL_DTYPE,
        scale_search: bool = True,
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
        effective_group_size = block_size if block_size is not None else group_size
        if effective_group_size < 16 or effective_group_size % 16 != 0:
            raise ValueError(
                f"W6A8 requires group_size >= 16 and divisible by 16, got {effective_group_size}"
            )
        if scale_dtype not in (torch.float8_e4m3fn, torch.float32):
            raise ValueError(f"W6A8 scale_dtype must be float8_e4m3fn or float32, got {scale_dtype}")

        self.convrot = True if convrot is None else bool(convrot)
        self.convrot_group_size = convrot_group_size
        self.group_size = effective_group_size
        self.block_size = effective_group_size
        self.dynamic_convrot = dynamic_convrot
        self.scale_dtype = scale_dtype
        self.scale_search = scale_search
        self.calib_scale = calib_scale
        self.has_bias = has_bias
        self.scale_optimization = scale_optimization
        self.optimizer_choice = optimizer
        self.last_group_size: Optional[int] = None
        self.last_quant_group_size: Optional[int] = None
        self._current_extra_tensors: Dict[str, torch.Tensor] = {}

    def _effective_scale(self, s_rel: torch.Tensor, s_channel: torch.Tensor) -> torch.Tensor:
        """Compute full float32 elementwise effective scale [N, K] = s_rel * s_channel."""
        return (
            s_rel.to(device=self.device, dtype=COMPUTE_DTYPE).repeat_interleave(self.group_size, dim=1)
            * s_channel.to(device=self.device, dtype=COMPUTE_DTYPE).unsqueeze(1)
        )

    def _optimize_w6a8_adaround(
        self,
        W_rot: torch.Tensor,
        unsigned_codes: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        X_rot: torch.Tensor,
        Y_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply SVD-guided AdaRound (learned soft rounding) over rotated W6A8 weights,
        minimizing local activation reconstruction MSE using cached calibration data.
        """
        M, N = W_rot.shape

        # 1. Compute SVD components of rotated parameter matrix
        U_k, Vh_k, k = self._compute_svd_components(W_rot, verbose=True)

        # 2. Setup soft rounding parameters V
        eff_scale = self._effective_scale(s_rel, s_channel).clamp_min(1e-12)
        W_scaled = W_rot / eff_scale
        W_floor = W_scaled.floor()

        target = W_scaled - W_floor
        target = torch.clamp(target, min=1e-6, max=1.0 - 1e-6)

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
            init_W_q_rounded = (unsigned_codes.to(COMPUTE_DTYPE) - 32.0)
            init_W_rounded_dequant = init_W_q_rounded * eff_scale
            init_mse_rounded = torch.nn.functional.mse_loss(X_rot @ init_W_rounded_dequant.T, Y_ref)
            init_svd_rounded = torch.linalg.norm(U_k.T @ (init_W_rounded_dequant - W_rot) @ Vh_k.T)

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
            desc=f"    Optimizing (AdaRound-W6A8-{self.optimizer_choice}-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            if optimizer is not None:
                optimizer.zero_grad()

            temp = T_start + (T_end - T_start) * (i / self.num_iter)
            h_V = torch.sigmoid(V / temp)
            W_q = W_floor + h_V
            W_dequant = W_q * eff_scale

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

            loss_mse_scaled = loss_mse / max(init_mse_rounded.item(), 1e-12)
            if alpha_svd > 0 and init_svd_rounded.item() > 1e-8:
                loss_svd_scaled = loss_svd / init_svd_rounded.item()
            else:
                loss_svd_scaled = 0.0

            loss = loss_mse_scaled + 0.01 * loss_svd_scaled + 0.1 * loss_reg

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

            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

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

        # Discretize V strictly to W6A8 symmetric range [-31, 31] -> stored [1, 63]
        with torch.no_grad():
            final_h_V = torch.sigmoid(best_V)
            final_round_up = (final_h_V >= 0.5).to(COMPUTE_DTYPE)
            final_signed = (
                (W_floor + final_round_up)
                .clamp(-W6A8_LEVELS, W6A8_LEVELS)
                .round()
                .to(torch.int32)
            )
            final_unsigned = final_signed + 32
            verbose(
                f"    - Discretization audit: {best_converged_ratio * 100:.2f}% of parameters converged to strict boundaries."
            )

        self._cleanup_tensors(U_k, Vh_k, V, best_V, W_scaled, W_floor, eff_scale)
        del U_k, Vh_k, V, best_V, W_scaled, W_floor, eff_scale
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_unsigned, s_rel, s_channel

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        calibration_data: Optional[torch.Tensor] = None,
        has_bias: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert a 2D weight tensor to packed W6A8 ConvRot format.

        Returns:
            Tuple of (packed_qdata, s_rel, s_channel, dequantized_weight, extra_tensors):
            - packed_qdata: torch.int8 of shape (N, 3 * K // 4)
            - s_rel: torch.float8_e4m3fn or torch.float32 of shape (N, K // group_size)
            - s_channel: torch.float32 of shape (N,)
            - dequantized_weight: torch.float32 of shape (N, K) in original (un-rotated) basis
            - extra_tensors: Dict containing bias_correction if computed
        """
        layer_has_bias = self.has_bias if has_bias is None else has_bias
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        rows, K = W_float32.shape
        layer_convrot_gs = resolve_w6a8_convrot_group_size(
            K,
            requested_group_size=self.convrot_group_size,
            dynamic=self.dynamic_convrot,
        )
        if layer_convrot_gs is None:
            raise ValueError(
                f"W6A8 ConvRot requires in_features ({K}) divisible by a power of 4 >= 4."
            )
        validate_w6a8_shape(K, self.group_size, layer_convrot_gs)
        self.last_group_size = layer_convrot_gs
        self.last_quant_group_size = self.group_size

        if torch.all(W_float32 == 0):
            verbose("  - Tensor is all zeros, skipping optimization.")
            packed_zeros = torch.zeros((rows, (3 * K) // 4), dtype=TARGET_W6A8_DTYPE, device=self.device)
            s_rel_ones = torch.ones((rows, K // self.group_size), dtype=self.scale_dtype, device=self.device)
            s_channel_ones = torch.ones(rows, dtype=SCALE_DTYPE, device=self.device)
            dequant_zeros = torch.zeros_like(W_float32)
            return packed_zeros, s_rel_ones, s_channel_ones, dequant_zeros, {}

        # Apply group-wise Hadamard rotation
        H = build_hadamard(layer_convrot_gs, device=self.device, dtype=COMPUTE_DTYPE)
        W_rot = rotate_weight(W_float32, H, layer_convrot_gs)
        verbose(f"    - Applied ConvRot Hadamard rotation (convrot_groupsize={layer_convrot_gs}, group_size={self.group_size}).")

        # Prepare calibration activations if provided
        X_rot = None
        Y_ref = None
        if calibration_data is not None:
            X_rot, Y_ref, _ = prepare_calibration_data(
                W_rot,
                calibration_data,
                True,
                layer_convrot_gs,
                self.device,
                COMPUTE_DTYPE,
                calib_scale=self.calib_scale,
            )

        # Baseline quantization (simple 2-pass ALS + grid-aware scale search)
        chunk_rows = self._compute_chunk_rows(rows, K, step=1)
        if chunk_rows < rows:
            verbose(f"    - Processing W6A8 quantization in row chunks of {chunk_rows}/{rows} (budget={self.chunk_budget})...")
            unsigned_codes = torch.empty((rows, K), dtype=torch.int32, device=self.device)
            s_rel = torch.empty((rows, K // self.group_size), dtype=self.scale_dtype, device=self.device)
            s_channel = torch.empty(rows, dtype=torch.float32, device=self.device)
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                u_chunk, s_rel_chunk, s_ch_chunk = quantize_w6a8_grouped(
                    W_rot[r0:r1],
                    group_size=self.group_size,
                    scale_dtype=self.scale_dtype,
                    scale_search=self.scale_search,
                )
                unsigned_codes[r0:r1] = u_chunk
                s_rel[r0:r1] = s_rel_chunk
                s_channel[r0:r1] = s_ch_chunk
                del u_chunk, s_rel_chunk, s_ch_chunk
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            unsigned_codes, s_rel, s_channel = quantize_w6a8_grouped(
                W_rot,
                group_size=self.group_size,
                scale_dtype=self.scale_dtype,
                scale_search=self.scale_search,
            )

        # Learned AdaRound optimization path
        if not self.no_learned_rounding and self.num_iter > 0 and X_rot is not None:
            if self.scale_optimization == "dualround":
                verbose("    - Scale Optimization: DUALROUND (Pass 1)")
                unsigned_codes, s_rel, s_channel = self._optimize_w6a8_adaround(
                    W_rot, unsigned_codes, s_rel, s_channel, X_rot, Y_ref
                )

                verbose("    - Scale Optimization: Re-estimating dual scales based on Pass 1 output...")
                eff_scale_p1 = self._effective_scale(s_rel, s_channel)
                dequant_opt = (unsigned_codes.to(COMPUTE_DTYPE) - 32.0) * eff_scale_p1
                groups = K // self.group_size
                group_max_opt = dequant_opt.view(rows, groups, self.group_size).abs().amax(dim=-1)
                group_scale_opt = (group_max_opt / float(W6A8_LEVELS)).clamp(min=1e-8)
                s_channel = (dequant_opt.abs().amax(dim=-1) / 127.0).clamp(min=1e-8).to(SCALE_DTYPE)
                s_rel = (group_scale_opt / s_channel.unsqueeze(1)).float()
                if self.scale_dtype != torch.float32:
                    s_rel = s_rel.to(self.scale_dtype)
                s_rel = s_rel.contiguous()
                if chunk_rows < rows:
                    for r0 in range(0, rows, chunk_rows):
                        r1 = min(r0 + chunk_rows, rows)
                        u_chunk, _, _ = quantize_w6a8_grouped(
                            W_rot[r0:r1],
                            group_size=self.group_size,
                            scale_dtype=self.scale_dtype,
                            scale_search=self.scale_search,
                        )
                        unsigned_codes[r0:r1] = u_chunk
                        del u_chunk
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    unsigned_codes, _, _ = quantize_w6a8_grouped(
                        W_rot,
                        group_size=self.group_size,
                        scale_dtype=self.scale_dtype,
                        scale_search=self.scale_search,
                    )

                del eff_scale_p1, dequant_opt, group_max_opt, group_scale_opt
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                verbose("    - Scale Optimization: DUALROUND (Pass 2)")
                unsigned_codes, s_rel, s_channel = self._optimize_w6a8_adaround(
                    W_rot, unsigned_codes, s_rel, s_channel, X_rot, Y_ref
                )
            else:
                verbose("    - Applying SVD-guided AdaRound optimization for W6A8 ConvRot...")
                unsigned_codes, s_rel, s_channel = self._optimize_w6a8_adaround(
                    W_rot, unsigned_codes, s_rel, s_channel, X_rot, Y_ref
                )

        # Pack 6-bit codes (nibble plane + 2-bit high plane)
        packed = pack_int6_w6a8(unsigned_codes)

        # Dequantize in rotated space through the exact runtime INT8 grid
        dequantized_rotated = dequantize_w6a8_rotated(packed, s_rel, s_channel, group_size=self.group_size)

        # Un-rotate to return dequantized weights in the original basis
        dequantized = rotate_weight(dequantized_rotated, H, layer_convrot_gs)

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

        self._cleanup_tensors(W_float32, W_rot, X_rot, Y_ref, dequantized_rotated, H)

        return (
            packed,
            s_rel.to(device=self.device, dtype=self.scale_dtype),
            s_channel.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized,
            extra_tensors,
        )
