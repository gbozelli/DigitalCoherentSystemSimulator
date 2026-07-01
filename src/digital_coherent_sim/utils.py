from __future__ import annotations

import importlib
import pathlib
from functools import lru_cache

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


@lru_cache(maxsize=32)
def load_legacy_module(module_name: str):
    return importlib.import_module(f"{ROOT.name}.{module_name}")


def ensure_2d_complex(array: np.ndarray, n_pols: int = 2) -> np.ndarray:
    arr = np.asarray(array, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] != n_pols:
        arr = np.vstack([arr, np.zeros_like(arr[0])])
    return arr
