# src/utils/seed.py
"""
Utilities for fully reproducible experiments across Python, NumPy, and PyTorch.
- Sets global RNG seeds
- Toggles deterministic/cuDNN behavior
- Provides DataLoader worker seeding helpers

Usage
-----
from utils.seed import seed_everything, dataloader_seed_args

seed_everything(42)  # set global seeds and deterministic flags

# When constructing DataLoaders:
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, **dataloader_seed_args(42))
val_loader   = DataLoader(dataset, batch_size=64, shuffle=False, **dataloader_seed_args(42))
"""

from __future__ import annotations
import os
import random
from typing import Dict, Any

try:
    import numpy as np
except Exception:  # numpy optional at import-time in some envs
    np = None  # type: ignore

import torch


def _set_python_hash_seed(seed: int) -> None:
    # Makes hashing of Python objects deterministic across runs.
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    cap_cudnn_determinism: bool = True,
) -> Dict[str, Any]:
    """
    Set RNG seeds for Python, NumPy, and PyTorch. Optionally enforce
    deterministic algorithms (may reduce speed or raise warnings).

    Args:
        seed: base seed.
        deterministic: if True, request deterministic ops where possible.
        cudnn_benchmark: forwarded to torch.backends.cudnn.benchmark.
        cap_cudnn_determinism: if True, also set torch.backends.cudnn.deterministic.

    Returns:
        A dict of flags/availability for logging.
    """
    s = int(seed)
    _set_python_hash_seed(s)

    random.seed(s)
    if np is not None:
        np.random.seed(s)

    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    # cuDNN knobs (safe on non-CUDA setups; flags just exist but have no effect)
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        except Exception:
            pass
        if cap_cudnn_determinism:
            try:
                torch.backends.cudnn.deterministic = bool(deterministic)
            except Exception:
                pass

    # Enforce deterministic algorithms where supported.
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            # Prefer warn-only if available (PyTorch >=1.11), so unsupported ops
            # emit warnings instead of raising hard errors.
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[arg-type]
        except TypeError:
            # Older versions: no warn_only kwarg
            torch.use_deterministic_algorithms(True)

    return {
        "seed": s,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "deterministic_algorithms": bool(getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()),
    }


def dataloader_seed_args(seed: int) -> Dict[str, Any]:
    """
    Return kwargs for DataLoader to make worker RNG deterministic.

    Example:
        DataLoader(ds, batch_size=64, shuffle=True, **dataloader_seed_args(42))
    """
    base_seed = int(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(base_seed)
    return {"worker_init_fn": _seed_worker, "generator": g}


__all__ = ["seed_everything", "dataloader_seed_args"]
