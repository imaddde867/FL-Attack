from typing import Optional

import torch


def resolve_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
