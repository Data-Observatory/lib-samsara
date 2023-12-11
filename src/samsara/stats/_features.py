from typing import Union

import numpy as np

__all__ = ["features"]


def plus_minus(p: np.ndarray, px_plus_y: np.ndarray, px_minus_y: np.ndarray) -> tuple:
    if p.shape[0] != p.shape[1]:
        raise ValueError("p array is not square.")

    n = p.shape[0]

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


def features(array: np.ndarray, n_feats: int = 7) -> np.ndarray:
    fts = np.full((n_feats,), np.nan)

    maxv = len(array)
    k = np.arange(maxv)
    k2 = k**2
    tk = np.arange(2 * maxv)
    # tk2 = tk**2
    i, j = np.mgrid[:maxv, :maxv]
    ij = i * j
    i_j2_p1 = (i - j) ** 2
    i_j2_p1 += 1
    i_j2_p1 = 1.0 / i_j2_p1
    i_j2_p1 = i_j2_p1.ravel()

    p = array / float(array.sum())
    pravel = p.ravel()
    px = p.sum(0)
    py = p.sum(1)

    ux = np.dot(px, k)
    uy = np.dot(py, k)
    vx = np.dot(px, k2) - ux**2
    vy = np.dot(py, k2) - uy**2

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)

    px_plus_y = np.full(2 * maxv, fill_value=0, dtype=np.double)
    px_minus_y = np.full(maxv, fill_value=0, dtype=np.double)
    px_plus_y, px_minus_y = plus_minus(p, px_plus_y, px_minus_y)

    fts[0] = np.dot(pravel, pravel)  # 1. ASM
    fts[1] = np.dot(k2, px_minus_y)  # 2. Contrast
    fts[2] = correlation(sx, sy, ux, uy, pravel, ij)  # 3. Correlation
    fts[3] = vx  # 4. Variance
    fts[4] = np.dot(i_j2_p1, pravel)  # 5. Inverse Difference Moment
    fts[5] = np.dot(tk, px_plus_y)  # 6. Sum Average
    fts[6] = entropy(pravel)  # 9. Entropy

    return fts
