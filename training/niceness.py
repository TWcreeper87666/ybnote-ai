"""Keep long jobs from hogging the machine the user is also using: below-
normal process priority (inherited by multiprocessing workers on Windows)
and a cap on torch / BLAS threads."""

from __future__ import annotations

import os
import sys

DEFAULT_THREADS = 4


def be_nice(threads: int = DEFAULT_THREADS) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    if sys.platform == "win32":
        import ctypes

        below_normal = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
    else:
        try:
            os.nice(10)
        except OSError:
            pass
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
