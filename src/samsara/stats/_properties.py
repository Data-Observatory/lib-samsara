from typing import Union

import numpy as np

__all__ = ["correlation", "entropy", "plus_minus"]


def plus_minus(
    p: np.ndarray,
    px_plus_y: np.ndarray,
    px_minus_y: np.ndarray,
    advanced_idx: bool = False,
) -> tuple:
    if p.shape[0] != p.shape[1]:
        raise ValueError("p array is not square.")

    n = p.shape[0]

    if advanced_idx is True:
        for i in range(n):
            px_plus_y[i : i + n] += p[i, :]
            px_minus_y[abs(i - np.arange(n))] += p[i, :]
        return px_plus_y, px_minus_y

    # Without advanced indexing
    for i in range(n):
        for j in range(n):
            px_plus_y[i + j] += p[i, j]
            px_minus_y[abs(i - j)] += p[i, j]

    return px_plus_y, px_minus_y


def entropy(p: np.array) -> Union[int, float]:
    p = p.ravel()
    p1 = p.copy()
    p1 += p == 0
    return -np.dot(np.log2(p1), p)


def correlation(
    sx: Union[int, float],
    sy: Union[int, float],
    ux: Union[int, float],
    uy: Union[int, float],
    pravel: np.ndarray,
    ij: np.ndarray,
) -> Union[int, float]:
    if sx == 0.0 or sy == 0.0:
        return 1.0
    return (1.0 / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)
