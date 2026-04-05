import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, Iterable, Literal
import math

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Fast stochastic rounding implementation for half-precision tensors.
    """
    with torch.no_grad():
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)
        target.copy_(result.view(dtype=torch.float32))

# Newton-Schulz iteration coefficients for orthogonalization
# From https://kexue.fm/archives/11059
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]

def reshape_to_2d(grad):
    """Reshape a tensor to 2D for matrix operations."""
    dimcount = len(grad.shape)
    if dimcount > 2:
        grad_2d = grad.reshape(len(grad), -1)
    elif dimcount < 2:
        grad_2d = grad.reshape(1, -1)
    else:
        grad_2d = grad
    return grad_2d


@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    orig_dtype = M.dtype
    if ortho_dtype is not None:
        M = M.to(ortho_dtype)
    
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    
    # Pre-calculate Identity matrix for better performance
    I = torch.eye(M.shape[1], dtype=M.dtype, device=M.device)
    
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        # Faster normalization
        M = M / (torch.linalg.norm(M).clamp_min_(1e-8))
        A = M.T @ M
        # 5th order Newton-Schulz update
        M = M @ (a * I + b * A + c * A @ A)
    
    if transpose:
        M = M.T
    
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M


@torch.no_grad()
def sanger_update(X: torch.Tensor, V: torch.Tensor, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single step of Sanger's Rule (Generalized Oja's rule) for online PCA."""
    X_norm = X / X.norm().clamp_min(1e-8)
    Y = X_norm @ V
    V_update = X_norm.T @ Y - V @ torch.triu(Y.T @ Y)
    V_new = V + lr * V_update
    Y_new = X @ V_new
    return V_new, Y_new


@torch.no_grad()
def past_update(X: torch.Tensor, V: torch.Tensor, P: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch Projection Approximation Subspace Tracking (PAST) algorithm with KxK inversion.
    X: [N, D] input batch
    V: [D, K] current basis
    P: [K, K] inverse covariance matrix
    beta: forgetting factor (0 < beta <= 1)
    """
    # X: [N, D], V: [D, K] -> Y: [N, K]
    Y = X @ V
    
    # C = Y.T @ Y: [K, K]
    C = Y.T @ Y
    
    # We use the KxK formulation of the RLS update for the inverse covariance P.
    # P_new = (beta * P^-1 + C)^-1 = (I + (1/beta) * P @ C)^-1 @ (P / beta)
    # This avoids NxN inversion where N is the larger dimension (e.g. pixels or channels).
    K = V.shape[1]
    I_K = torch.eye(K, device=X.device, dtype=X.dtype)
    
    # denom = beta * I + P @ C
    # P_new = solve(beta * I + P @ C, P)
    # We enforce symmetry for numerical stability.
    P_new = torch.linalg.solve(beta * I_K + P @ C, P)
    P_new = (P_new + P_new.T) * 0.5
    
    # Update basis V: [D, K]
    # V_new = V + (X.T @ Y - V @ C) @ P_new
    # This avoids explicit creation of large [D, N] residuals.
    V_new = V + (X.T @ Y - V @ C) @ P_new
    
    Y_new = X @ V_new
    return V_new, Y_new, P_new


@torch.no_grad()
def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, eps=1e-16):
    """
    CAME-style factorized denominator computation.
    Combines row-wise and column-wise variance factors.
    Uses in-place operations for memory efficiency.
    """
    # Row factor with epsilon to prevent division by zero
    r_factor = (exp_avg_sq_row + eps)
    r_factor.sqrt_()  # In-place sqrt
    r_factor.unsqueeze_(-1)  # In-place unsqueeze
    
    # Column factor with epsilon
    c_factor = ((exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=0, keepdim=True) + eps)).unsqueeze(-2)
    c_factor.sqrt_()  # In-place sqrt
    
    # Combine with broadcasting support
    return torch.mul(r_factor, c_factor)


class WiwiOpt(Optimizer):
    r"""
    WiwiOpt (V1.3 with CAME-style factorization).

    A gradient descent optimizer that combines several stabilization & acceleration techniques to produce
    high-signal stable parameter updates.

    WiwiOpt works by:
    1. RMS-based gradient normalization: Incoming gradients are normalized
       by a polynomial-decay EMA of their per-row RMS, preventing exploding
       or vanishing gradient magnitudes.
    2. Egalitarian Gradient Descent (EGD) preconditioning: For 2D+
       parameters, a low-rank SVD approximation is used to precondition the
       gradient, equalizing contribution across singular directions.
    3. Polynomial-schedule momentum: Momentum and accumulation use
       polynomial schedules instead of fixed betas, providing
       smoothing that naturally increases over early training.
    4. Newton-Schulz orthogonalization (Muon): The effective gradient is
       orthogonalized via Newton-Schulz iteration for multi-dimensional
       parameters, producing direction-pure updates.
    5. NorMuon scaling: After orthogonalization, the update is re-scaled
       using a tracked second-moment estimate to maintain consistent update
       magnitudes, then re-projected to preserve the original norm.
    6. Projection re-scaling: The orthogonalized step is re-scaled by its
       projection onto the un-orthogonalized effective gradient, preserving
       meaningful magnitude information.
    7. Cautious masking: Updates are masked so that only components
       agreeing in sign with the raw gradient are kept, preventing
       counterproductive steps.
    8. Dynamic learning rate: Per-parameter learning rate adjustment based
       on the alignment between the EMA of parameter deltas and the EMA of
       their norms, optionally boosted by an ``atan2``-based scaling factor.
    9. CAME-style factorized variance tracking: Uses row-wise AND column-wise
       variance estimates (like CAME optimizer) for more accurate gradient
       normalization, with in-place operations for memory efficiency.

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float, float] or Tuple[float, float]):
            Exponents for the de-biased beta schedules.
            ``beta1`` controls momentum and gradient accumulation decay,
            ``beta2`` controls the variance tracker and NorMuon second-moment
            decay, and ``beta3`` controls the dynamic learning rate EMAs.
            (default: (0.95, 0.995, 0.99)).
        eps (float): Numerical stability term for divisions and clamps
            (default: 1e-16).
        weight_decay (float): Decoupled weight decay coefficient
            (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied,
            weight_decay * weight_decay_rate**step
            (default: 1.0).
        normuon (bool): Apply NorMuon second-moment scaling after
            orthogonalization to stabilize update magnitudes
            (default: True).
        use_compile (bool): Use ``torch.compile`` on the orthogonalization
            and SVD functions for faster execution (default: True).
        ortho_dtype (str or None): Data type for Newton-Schulz
            orthogonalization. Accepts ``None`` (defaults to float32) or a
            string like ``"torch.bfloat16"`` (default: None).
        stochastic_fp (bool): Use stochastic rounding when parameters are
            stored in bfloat16, reducing quantization bias (default: True).
        dynamic_lr (bool): Enable per-row dynamic learning rate
            adjustment based on delta alignment (default: True).
        dynamic_lr_boost (bool): When ``dynamic_lr`` is enabled, apply an
            additional ``atan2``-based boost factor that amplifies the
            learning rate when parameter deltas are large relative to their
            directional EMA (default: True).
        egd (bool): Enable Egalitarian Gradient Descent preconditioning via
            low-rank SVD for parameters with 2+ dimensions, equalizing
            gradient contribution across singular directions
            (default: True).
        egd_oja (bool): Enables a lightweight approximation
            of EGD using Sanger's rule (Generalized Oja's rule) in place 
            of full SVD tracking (default: True).
        egd_method (str): Method for online decomposition tracking.
            Accepts 'past' (default), 'oja', or 'svd'.
            If 'svd' is used, `egd_oja` is ignored.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        betas: Tuple[float, float, float] = (0.95, 0.995, 0.99),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 1.0,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: Optional[torch.dtype] = None,
        stochastic_fp: bool = True,
        dynamic_lr: bool = True,
        dynamic_lr_boost: bool = True,
        egd: bool = True, 
        egd_oja: bool = True,
        egd_method: Literal['past', 'oja', 'svd'] = 'past',
        use_poly_betas: bool = True,
        use_muon: bool = True,
    ):
        if len(betas) == 2:
            betas = (betas[0], betas[0], betas[1])
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if ortho_dtype is None:
            ortho_dtype = torch.bfloat16
        elif isinstance(ortho_dtype, str):
            dtype_name = ortho_dtype.split('.')[-1]
            ortho_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_rate=weight_decay_rate,
            normuon=normuon,
            use_compile=use_compile,
            ortho_dtype=ortho_dtype,
            stochastic_fp=stochastic_fp,
            dynamic_lr=dynamic_lr,
            dynamic_lr_boost=dynamic_lr_boost,
            egd=egd,
            egd_oja=egd_oja,
            egd_method=egd_method,
            use_poly_betas=use_poly_betas,
            use_muon=use_muon,
        )
        self.ortho_func = torch.compile(orthogonalize) if use_compile else orthogonalize
        self.oja_func = None
        self.past_func = None
        self.svd_func = None
        if egd:
            if egd_method == 'oja' or (egd_method is None and egd_oja):
                self.oja_func = torch.compile(sanger_update) if use_compile else sanger_update
            elif egd_method == 'past':
                self.past_func = torch.compile(past_update) if use_compile else past_update
            elif egd_method == 'svd' or (egd_method is None and not egd_oja):
                self.svd_func = torch.compile(torch.svd_lowrank) if use_compile else torch.svd_lowrank
        
        if use_compile:
            try:
                import torch._inductor.config as inductor_config
                # As suggested by the warning to handle many distinct shapes
                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
                # Silence the warning about recording too many CUDAGraphs
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
            except (ImportError, AttributeError):
                pass
        
        super(WiwiOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if len(group['betas']) == 2:
                beta1, beta2, beta3 = group['betas'][0], group['betas'][0], group['betas'][1]
            else:
                beta1, beta2, beta3 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            weight_decay_rate = group['weight_decay_rate']
            stochastic_fp = group['stochastic_fp']
            egd = group['egd']
            egd_oja = group['egd_oja']
            egd_method = group['egd_method']
            dynamic_lr = group['dynamic_lr']
            dynamic_lr_boost = group['dynamic_lr_boost']
            use_poly_betas = group['use_poly_betas']
            use_muon = group['use_muon']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('WiwiOpt does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['accum'] = torch.ones_like(p.mean(dim=-1, keepdim=True), memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Factorized variance tracking for 2D+ tensors (CAME-style)
                    if p.ndim >= 2:
                        state['exp_avg_sq_row'] = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
                        col_shape = p.shape[:-2] + p.shape[-1:]
                        state['exp_avg_sq_col'] = torch.zeros(col_shape, device=p.device, dtype=p.dtype)
                    else:
                        # 1D tensors: fall back to full variance tracking
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if dynamic_lr:
                        state['delta_ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['delta_norm_ema'] = torch.zeros_like(p.mean(dim=-1, keepdim=True), memory_format=torch.preserve_format)
                    if p.ndim >= 1 and group["normuon"]:
                        grad_2d = reshape_to_2d(grad)
                        state['normuon_second_momentum'] = torch.zeros(grad_2d.shape[0], 1, device=p.device, dtype=p.dtype)

                state['step'] += 1
                step = state['step']

                accum = state['accum']
                exp_avg = state['exp_avg']
                
                # Mixed precision handling
                use_stochastic = stochastic_fp and p.dtype in {torch.bfloat16}
                
                # Initialize variables to avoid unbound errors
                p_work = p.detach()
                grad_work = grad.detach()
                accum_work = accum.detach()
                exp_avg_work = exp_avg.detach()
                
                # Initialize factorized state work variables
                exp_avg_sq_row_work = None
                exp_avg_sq_col_work = None
                exp_avg_sq_work = None
                
                delta_ema_work = None
                delta_norm_ema_work = None
                normuon_z = None
                
                if use_stochastic:
                    p_work = p_work.to(torch.float32)
                    grad_work = grad_work.to(torch.float32)
                    accum_work = accum_work.to(torch.float32)
                    exp_avg_work = exp_avg_work.to(torch.float32)
                    
                # Handle factorized states for mixed precision
                if p.ndim >= 2:
                    exp_avg_sq_row_work = state['exp_avg_sq_row'].detach()
                    exp_avg_sq_col_work = state['exp_avg_sq_col'].detach()
                    if use_stochastic:
                        exp_avg_sq_row_work = exp_avg_sq_row_work.to(torch.float32)
                        exp_avg_sq_col_work = exp_avg_sq_col_work.to(torch.float32)
                else:
                    exp_avg_sq_work = state['exp_avg_sq'].detach()
                    if use_stochastic:
                        exp_avg_sq_work = exp_avg_sq_work.to(torch.float32)
                    
                if dynamic_lr:
                    delta_ema_work = state['delta_ema'].detach()
                    delta_norm_ema_work = state['delta_norm_ema'].detach()
                    if use_stochastic:
                        delta_ema_work = delta_ema_work.to(torch.float32)
                        delta_norm_ema_work = delta_norm_ema_work.to(torch.float32)
                        
                if p.ndim >= 1 and group["normuon"]:
                    normuon_z = state['normuon_second_momentum'].detach()
                    if use_stochastic:
                        normuon_z = normuon_z.to(torch.float32)

                if use_poly_betas:
                    poly_beta1 = ((beta1**(step) - beta1) / (beta1**(step) - 1.0))
                    poly_beta2 = ((beta2**(step) - beta2) / (beta2**(step) - 1.0))
                    poly_beta3 = ((beta3**(step) - beta3) / (beta3**(step) - 1.0))
                else:
                    poly_beta1, poly_beta2, poly_beta3 = beta1, beta2, beta3

                grad_rms = grad_work.pow(2).mean(dim=-1, keepdim=True)
                accum_work.lerp_(grad_rms, 1. - poly_beta1)

                grad_work.div_(accum_work.sqrt().clamp_min_(eps)).clamp_(-step, step)

                if egd and p_work.ndim >= 2:
                    grad_work_2d = reshape_to_2d(grad_work)
                    m_dim, n_dim = grad_work_2d.size(0), grad_work_2d.size(1)
                    current_rank = min(128, m_dim, n_dim)
                    
                    if current_rank > 0:
                        is_online = (egd_method in ['oja', 'past']) or (egd_method is None and egd_oja)
                        if is_online:
                            if 'oja_basis' not in state:
                                track_u = (m_dim < n_dim)
                                feature_dim = m_dim if track_u else n_dim
                                # Always use float32 for online PCA states to ensure numerical stability.
                                basis = torch.randn(feature_dim, current_rank, device=p_work.device, dtype=torch.float32)
                                basis, _ = torch.linalg.qr(basis)
                                state['oja_basis'] = basis
                                if egd_method == 'past':
                                    state['inv_cov'] = torch.eye(current_rank, device=p_work.device, dtype=torch.float32) * 0.1
                                
                            track_u = (m_dim < n_dim)
                            oja_basis_work = state['oja_basis']
                            # Ensure we work in float32 for the online update
                            oja_basis_work = oja_basis_work.detach().float()
                                
                            X_for_oja = grad_work_2d.T if track_u else grad_work_2d
                            X_for_oja = X_for_oja.float()
                                
                            try:
                                Y_new = None
                                if egd_method == 'past' and self.past_func is not None:
                                    inv_cov_work = state['inv_cov'].detach().float()
                                    
                                    # Use a stable forgetting factor for PAST.
                                    # It should match the EMA factor (poly_beta1) but clamped for stability.
                                    past_beta = max(poly_beta1, 0.99)
                                    oja_basis_work, Y_new, inv_cov_work = self.past_func(X_for_oja, oja_basis_work, inv_cov_work, past_beta)
                                    
                                    state['inv_cov'].copy_(inv_cov_work)
                                elif egd_method == 'oja' and self.oja_func is not None:
                                    oja_basis_work, Y_new = self.oja_func(X_for_oja, oja_basis_work, 1. - poly_beta1)
                                
                                if Y_new is not None:
                                    # Normalize the basis vectors and the projections to ensure the 
                                    # preconditioned gradient magnitude is stable regardless of basis drift.
                                    basis_norm = oja_basis_work / oja_basis_work.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    proj_norm = Y_new / Y_new.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    
                                    if track_u:
                                        grad_precond = basis_norm @ proj_norm.T
                                    else:
                                        grad_precond = proj_norm @ basis_norm.T
                                        
                                    state['oja_basis'].copy_(oja_basis_work)
                                    grad_work = grad_precond.to(p_work.dtype).view_as(p_work)
                            except RuntimeError:
                                pass
                        else:
                            try:
                                # Use float32 for SVD stability if it was half precision
                                dtype_orig = grad_work_2d.dtype
                                grad_f32 = grad_work_2d.float()
                                
                                if self.svd_func is not None:
                                    U, S, _ = self.svd_func(grad_f32, q=current_rank)
                                    
                                    U = U.to(dtype_orig)
                                    S = S.to(dtype_orig)
                                    
                                    S = torch.maximum(S, torch.tensor(eps, device=S.device, dtype=S.dtype))
                                    S_inv = 1.0 / S
                                    
                                    aux = (U * S_inv.unsqueeze(0)) @ U.mT
                                    grad_precond = aux @ grad_work_2d
                                    
                                    grad_work = grad_precond.view_as(p_work)
                            except RuntimeError:
                                # Fallback if SVD fails to converge (rare)
                                pass

                # CAME-style factorized variance tracking with in-place operations
                if p_work.ndim >= 2:
                    grad_err = grad_work - exp_avg_work
                    grad_err.pow_(2)  # In-place square
                    
                    # Update row-wise variance (mean over last dimension)
                    exp_avg_sq_row_work.lerp_(grad_err.mean(dim=-1), weight=1. - poly_beta2)
                    
                    # Update column-wise variance (mean over second-to-last dimension)
                    if grad_err.ndim > 2:
                        exp_avg_sq_col_work.lerp_(grad_err.mean(dim=-2), weight=1. - poly_beta2)
                    else:
                        exp_avg_sq_col_work.lerp_(grad_err.mean(dim=0), weight=1. - poly_beta2)
                    
                    # Compute factorized denominator
                    denom = _approx_sq_grad(exp_avg_sq_row_work, exp_avg_sq_col_work, eps)
                else:
                    # 1D tensors: use simple variance tracking
                    grad_err = grad_work - exp_avg_work
                    grad_err.pow_(2)  # In-place
                    exp_avg_sq_work.lerp_(grad_err, weight=1. - poly_beta2)
                    denom = exp_avg_sq_work.sqrt_().clamp_min_(eps)  # In-place sqrt

                # Momentumize with in-place operations
                exp_avg_work.lerp_(grad_work, weight=1. - poly_beta1)
                
                # Compute effective gradient
                g_eff_mom = grad_work.clone()
                g_eff_mom.lerp_(exp_avg_work, weight=poly_beta1)
                g_eff_mom.div_(denom)  # Apply factorized denominator in-place

                if p_work.ndim >= 1:
                    if use_muon:
                        full_step_2d = reshape_to_2d(g_eff_mom)
                        
                        # Newton-Schulz Orthogonalization (Muon)
                        Q = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                        # NorMuon update & re-norm
                        if group["normuon"] and normuon_z is not None:
                            vnorm = Q.norm(dim=(-2, -1), keepdim=True)

                            v_mean = torch.mean(Q * Q, dim=-1, keepdim=True)
                            normuon_z.lerp_(v_mean, 1 - poly_beta2)
                            step_size = normuon_z.sqrt().clamp_min_(eps)
                            Q.div_(step_size)

                            vnorm_new = Q.norm(dim=(-2, -1), keepdim=True)
                            Q = Q * (vnorm / vnorm_new.clamp_min(eps))

                        final_step = Q.view_as(p_work)

                        # Re-scaling: final_step functionally sums to 1.
                        # We re-scale it to the magnitude of the projection onto the un-orthogonalized effective gradient
                        scale_factor = (g_eff_mom * final_step).sum()
                        final_step.mul_(scale_factor)
                    else:
                        final_step = g_eff_mom
                else:
                    final_step = g_eff_mom

                # Cautious masking
                scale_factor_mask = (grad_work * final_step > 0).to(final_step.dtype)
                mask_mean = scale_factor_mask.mean().clamp_min_(1e-3)
                scale_factor_mask.div_(mask_mean)
                final_step.mul_(scale_factor_mask)

                # Dynamic Learning Rate Adjustment
                lr_adj = torch.ones_like(p.mean())
                if dynamic_lr and delta_ema_work is not None and delta_norm_ema_work is not None:
                    if step > 1:
                        # True norm of EMA of deltas vs EMA of accumulated norms of deltas
                        alignment_ratio = delta_ema_work.norm(dim=-1, keepdim=True) / delta_norm_ema_work.clamp_min(eps)
                        # Parameter-wise update scaling
                        if dynamic_lr_boost:
                            update_ratio = delta_norm_ema_work.atan2(delta_ema_work.abs()).mul_(1.27323954474)
                            lr_adj = alignment_ratio * update_ratio
                        else:
                            lr_adj = alignment_ratio
                    else:
                        lr_adj = torch.ones_like(p.mean())
                        
                    final_step.mul_(lr_adj)
                    
                    # Update EMAs
                    current_norm = final_step.norm(dim=-1, keepdim=True)
                    delta_ema_work.lerp_(final_step, 1. - poly_beta3)
                    delta_norm_ema_work.lerp_(current_norm, 1. - poly_beta3)

                # Apply Update
                if weight_decay != 0:
                    weight_decay_multiplier = weight_decay_rate**step
                    p_mid = torch.where(p_work * final_step > 0, p_work, torch.zeros_like(p_work))
                    p_work.add_(p_mid * lr_adj if dynamic_lr else p_mid, alpha=-lr * weight_decay * weight_decay_multiplier)
                
                p_work.add_(final_step, alpha=-lr)

                # State Sync
                if use_stochastic:
                    copy_stochastic_(accum, accum_work)
                    copy_stochastic_(exp_avg, exp_avg_work)
                    # Sync factorized variance states
                    if p.ndim >= 2:
                        copy_stochastic_(state['exp_avg_sq_row'], exp_avg_sq_row_work)
                        copy_stochastic_(state['exp_avg_sq_col'], exp_avg_sq_col_work)
                    else:
                        copy_stochastic_(state['exp_avg_sq'], exp_avg_sq_work)
                    if dynamic_lr and delta_ema_work is not None and delta_norm_ema_work is not None:
                        copy_stochastic_(state['delta_ema'], delta_ema_work)
                        copy_stochastic_(state['delta_norm_ema'], delta_norm_ema_work)
                    copy_stochastic_(p, p_work)
                    if p.ndim >= 1 and group["normuon"] and normuon_z is not None:
                        copy_stochastic_(state['normuon_second_momentum'], normuon_z)
                else:
                    accum.copy_(accum_work)
                    exp_avg.copy_(exp_avg_work)
                    # Sync factorized variance states
                    if p.ndim >= 2:
                        state['exp_avg_sq_row'].copy_(exp_avg_sq_row_work)
                        state['exp_avg_sq_col'].copy_(exp_avg_sq_col_work)
                    else:
                        state['exp_avg_sq'].copy_(exp_avg_sq_work)
                    if dynamic_lr:
                        state['delta_ema'].copy_(delta_ema_work)
                        state['delta_norm_ema'].copy_(delta_norm_ema_work)
                    p.copy_(p_work)
                    if p.ndim >= 1 and group["normuon"]:
                        state['normuon_second_momentum'].copy_(normuon_z)

        return loss
