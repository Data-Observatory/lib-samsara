"""Subpackage for statistics from gray level co-occurrence matrix (`samsara.stats.glcm`)
"""
import math
from itertools import product
from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
from skimage.feature import graycomatrix

from ._features import correlation, entropy, plus_minus


def glcm(
    array: xr.DataArray, radius: int, n_feats: int = 7, **kwargs
) -> Union[da.Array, np.ndarray]:
    """Calculate texture properties of an image.

    For each pixel in the image, it uses a window of values surrounding it and calculates the glcm
    and the properties of that glcm. The properties are the following:
    - ASM
    - Contrast
    - Correlation
    - Variance
    - Inverse Difference Moment
    - Sum Average
    - Entropy

    Parameters
    ----------
    array : xr.DataArray
        2-dim DataArray image.
    radius : int
        Radius of the moving window.
    n_feats : int, optional
        Number of features or properties computed, by default 7.
    kwargs :
        Other keyword arguments to pass to function :func:`textures <samsara.stats.glcm.textures>`.

    Returns
    -------
    Union[da.Array, np.ndarray]
        3-dim array with the texture properties for each pixel. The new axis is located at the first
        dimension, and indexes the property.

    Raises
    ------
    ValueError
        If array is not 2-dimensional.

    See Also
    --------
    :func:`textures <samsara.stats.glcm.textures>`
    :func:`graycomatrix <skimage.feature.graycomatrix>`

    """
    data = array.data

    if data.ndim != 2:
        raise ValueError(f"Expected 2-dimensional data, got {data.ndim} dimensions")

    kwargs["radius"] = radius
    kwargs["n_feats"] = n_feats

    if isinstance(data, da.Array):
        chunks_ = ((n_feats,), *list(data.chunks))
        glcm_cube = da.map_overlap(
            textures,
            data,
            depth=radius,
            boundary=0,
            trim=False,
            align_arrays=False,
            allow_rechunk=False,
            dtype=float,
            chunks=chunks_,
            drop_axis=None,
            new_axis=0,
            **kwargs,
        )
        return glcm_cube
    # Non-chunked
    data_pad = np.pad(data, ((radius, radius), (radius, radius)))
    glcm_cube = textures(data_pad, radius, **kwargs)
    return glcm_cube


def matrix(
    array: np.ndarray,
    distances: np.ndarray,
    angles: np.ndarray,
    levels: Union[int, None] = None,
    symmetric: bool = False,
    normed: bool = False,
    nan_supression: int = 0,
    rescale_normed: bool = False,
) -> np.ndarray:
    """Calculate the gray level co-occurrence matrix.

    Parameters
    ----------
    array : np.ndarray
        2-dimensional array. Input image.
    distances : np.ndarray
        List of pixel pair distance offsets.
    angles : np.ndarray
        List of pixel pair angles in radians.
    levels : Union[int, None], optional
        Number of gray-levels counted, by default None. This argument is required for 16-bit images
        or higher and is typically the maximum of the image.
    symmetric : bool, optional
        Whether or not the output is symmetric, by default False.
    normed : bool, optional
        Whether or not to normalize each offset matrix, by default False.
    nan_supression : int, optional
        Method used to replace values in each glcm, by default 0.

        - 0
            Do nothing to glcm.
        - 1
            Position (0,0) in the glcm is replaced with 0s.
        - 2
            Row and column 0 in the glcm are replaced with 0s.
        - 3
            Row and column 0 are removed from the glcm. Only works for `levels` + 1.
        - other
            Do nothing to glcm.

    rescale_normed : bool, optional
        Whether to rescale the resulting gray level co-occurrence matrix so the elements sum to 1,
        even after `nan_supression`, by default False.

    Returns
    -------
    np.ndarray
        4-dimensional array. The gray level co-occurrence matrix/histogram.

    See Also
    --------
    :func:`graycomatrix <skimage.feature.graycomatrix>`
    """
    glcm_ = graycomatrix(
        array, distances, angles, levels=levels, symmetric=symmetric, normed=normed
    )
    if nan_supression == 1:
        glcm_[0, 0, :, :] = 0
    if nan_supression == 2:
        glcm_[:, 0, :, :] = 0
        glcm_[0, :, :, :] = 0
    if nan_supression == 3:
        glcm_ = glcm_[1:, 1:, :, :]
    if rescale_normed is True:
        glcm_ = glcm_ / np.sum(glcm_, axis=(0, 1))
    return glcm_


def features(array: np.ndarray, n_feats: int = 7) -> np.ndarray:
    """Calculate texture features of a gray level co-occurrence matrix.

    From a gray level co-occurrence matrix compute the following properties:
    - ASM
    - Contrast
    - Correlation
    - Variance
    - Inverse Difference Moment
    - Sum Average
    - Entropy

    Parameters
    ----------
    array : np.ndarray
        4-dimensional array. Gray level co-occurrence histogram of an image. The coordinates are
        (levels, levels, number of distances, number of angles).
    n_feats : int, optional
        Number of features or properties computed with the glcm, by default 7.

    Returns
    -------
    np.ndarray
        1-dim array of length `n_feats` with the features or properties computed from `array`.

    See Also
    --------
    :func:`matrix <samsara.stats.glcm.matrix>`
    :func:`graycomatrix <skimage.feature.graycomatrix>`
    """
    fts = np.full((n_feats,), np.nan)

    maxv = len(array)
    k = np.arange(maxv)
    k2 = k**2
    tk = np.arange(2 * maxv)

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

    fts = [
        np.dot(pravel, pravel),  # 1. ASM
        np.dot(k2, px_minus_y),  # 2. Contrast
        correlation(sx, sy, ux, uy, pravel, ij),  # 3. Correlation
        vx,  # 4. Variance
        np.dot(i_j2_p1, pravel),  # 5. Inverse Difference Moment
        np.dot(tk, px_plus_y),  # 6. Sum Average
        entropy(pravel),  # 9. Entropy
    ]

    if len(fts) < n_feats:
        return np.pad(
            np.array(fts, dtype=float), (0, n_feats - len(fts)), constant_values=np.nan
        )

    return fts[:n_feats]


def textures(
    array: np.array,
    radius: int = 1,
    n_feats: int = 7,
    nan_supression: int = 0,
    skip_nan: bool = True,
    rescale_normed: bool = False,
    distances: Union[list, None] = None,
    angles: Union[list, None] = None,
    **kwargs,
) -> np.ndarray:
    """Calculate texture properties of an in-memory image.

    For each pixel in the image, it uses a window of values surrounding it and calculates the glcm
    and the properties of that glcm. The properties are the following:
    - ASM
    - Contrast
    - Correlation
    - Variance
    - Inverse Difference Moment
    - Sum Average
    - Entropy

    Parameters
    ----------
    array : np.array
        2-dim image.
    radius : int, optional
        Radius of the moving window, by default 1.
    n_feats : int, optional
        Number of features or properties computed, by default 7.
    nan_supression : int, optional
        Method used to replace values in each glcm, by default 0.

        - 0
            Do nothing to glcm.
        - 1
            Position (0,0) in the glcm is replaced with 0s.
        - 2
            Row and column 0 in the glcm are replaced with 0s.
        - 3
            Row and column 0 are removed from the glcm. Only works for `levels` + 1.
        - other
            Do nothing to glcm.

    skip_nan : bool, optional
        Whether or not to replace fill NaN the texture of a pixel if its original value (in `array`)
        is 0, by default True.
    rescale_normed : bool, optional
        Whether to rescale the resulting gray level co-occurrence matrix so the elements sum to 1,
        even after `nan_supression`, by default False.
    distances : Union[list, None], optional
        List of pixel pair distance offsets, by default [-1, 0, 1, 2].
    angles : Union[list, None], optional
        List of pixel pair angles in radians, by default [0, :math:`$\\pi/2$`].
    kwargs :
        Other keywords arguments to pass to function
        :func:`graycomatrix <skimage.feature.graycomatrix>`.

    Returns
    -------
    np.ndarray
        3-dim array with the texture properties for each pixel. The new axis is located at the first
        dimension, and indexes the property.

    See Also
    --------
    :func:`matrix <samsara.stats.glcm.matrix>`
    :func:`graycomatrix <skimage.feature.graycomatrix>`
    """
    if distances is None:
        distances = range(-1, 2)
    if angles is None:
        angles = [0, math.pi / 2]

    view = np.lib.stride_tricks.sliding_window_view(
        array, (radius * 2 + 1, radius * 2 + 1)
    )

    response = np.zeros([n_feats, *list(view.shape[:2])])
    range_i, range_j = range(view.shape[0]), range(view.shape[1])

    for i, j in product(range_i, range_j):
        subarray = view[i, j, :, :]

        if array[i + radius, j + radius] == 0 and skip_nan:
            response[:, i, j] = np.repeat(np.nan, n_feats)
        else:
            glcm_ = matrix(
                subarray,
                distances,
                angles,
                nan_supression=nan_supression,
                rescale_normed=rescale_normed,
                **kwargs,
            )

            response[:, i, j] = features(glcm_, n_feats=n_feats)

    if nan_supression > 0:
        sub_array = array[
            radius : (array.shape[0] - radius), radius : (array.shape[1] - radius)
        ]
        response[
            (np.repeat(sub_array[np.newaxis, :, :], n_feats, axis=0)) == 0
        ] = np.nan

    return response
