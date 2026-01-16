"""
Pinned memory utilities for faster CPU→GPU tensor transfers.

Pinned (page-locked) memory enables faster DMA transfers to GPU.
Uses PyTorch's native pin_memory() with non_blocking transfers.
"""
import torch
from typing import Optional

# Module-level configuration
_verbose = False
_pinned_transfer_stats = {"pinned": 0, "fallback": 0}

def set_verbose(enabled: bool):
    """Enable/disable verbose output for pinned transfers."""
    global _verbose
    _verbose = enabled

def get_pinned_transfer_stats():
    """Return pinned transfer statistics for verification."""
    return _pinned_transfer_stats.copy()

def reset_pinned_transfer_stats():
    """Reset transfer statistics."""
    global _pinned_transfer_stats
    _pinned_transfer_stats = {"pinned": 0, "fallback": 0}

def transfer_to_gpu_pinned(
    tensor: torch.Tensor,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Transfer tensor to GPU using pinned memory for faster transfer."""
    global _pinned_transfer_stats

    # Skip if not a CPU tensor or CUDA unavailable
    if tensor.device.type != 'cpu' or not torch.cuda.is_available():
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    # Skip if target is not CUDA
    if not str(device).startswith('cuda'):
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    try:
        pinned = tensor.pin_memory()

        if dtype is not None:
            result = pinned.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result = pinned.to(device=device, non_blocking=True)

        torch.cuda.current_stream().synchronize()

        # One-time confirmation on first success
        if _pinned_transfer_stats["pinned"] == 0:
            print("  [pinned_transfer] Pinned memory active - faster GPU transfers enabled")

        _pinned_transfer_stats["pinned"] += 1
        if _verbose:
            print(f"  [pinned_transfer] Pinned: {tensor.shape} ({tensor.numel() * tensor.element_size() / 1024:.1f} KB)")

        return result

    except Exception as e:
        _pinned_transfer_stats["fallback"] += 1
        if _verbose:
            print(f"  [pinned_transfer] Fallback: {e}")

        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
