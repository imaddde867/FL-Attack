"""Device utilities for PyTorch backend selection."""

from typing import Optional
import torch


def resolve_device(requested: Optional[str] = None) -> str:
    """
    Resolve the best available compute device.
    
    Args:
        requested: Explicitly requested device ('cuda', 'mps', 'cpu').
                   If None, auto-detects best available.
    
    Returns:
        Device string suitable for torch operations.
    """
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
