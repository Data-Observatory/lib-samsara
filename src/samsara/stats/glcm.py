"""Subpackage for statistics from gray level co-occurrence matrix (`samsara.stats.glcm`)
"""

import math
from itertools import product
from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
from skimage.feature import graycomatrix

from ._properties import correlation, entropy, plus_minus


def glcm_textures(
    array: xr.DataArray, radius: int, n_feats: int = 7, **kwargs
) -> xr.DataArray:
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
        - Difference variance
        - Dissimilarity

    Parameters
    ----------
    array : xr.DataArray
        2-dim DataArray image.
    radius : int
        Radius of the moving window.
    n_feats : int, optional
        Number of features or properties computed, by default 7.
    kwargs :
        Other keyword arguments to pass to function
        :func:`block_glcm_textures <samsara.stats.glcm.block_glcm_textures>`.

    Returns
    -------
    xr.DataArray
        3-dim array with the texture properties for each pixel. The new axis is located at the first
        dimension, and indexes the property.

    Raises
    ------
    ValueError
        If `array` is not 2-dimensional.
    TypeError
        If the data in `array` is neither da.Array or np.ndarray.

    See Also
    --------
    :func:`block_glcm_textures <samsara.stats.glcm.block_glcm_textures>`
    :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`

    """
    data = array.data

    if data.ndim != 2:
        raise ValueError(f"Expected 2-dimensional data, got {data.ndim} dimensions")

    if not isinstance(data, (da.Array, np.ndarray)):
        raise TypeError(
            "Invalid type of data array in input. Expected either dask.Array or numpy.ndarray, got"
            f" {type(data)}."
        )

    # Save coords and attrs
    array_coords = array.coords.copy()
    array_attrs = array.attrs.copy()
    array_dims = array.dims

    kwargs["radius"] = radius
    kwargs["n_feats"] = n_feats

    if isinstance(data, da.Array):
        chunks_ = ((n_feats,), *list(data.chunks))
        glcm_cube = da.map_overlap(
            block_glcm_textures,
            data,
            depth=radius,
            boundary="reflect",
            trim=False,
            align_arrays=False,
            allow_rechunk=False,
            dtype=float,
            chunks=chunks_,
            drop_axis=None,
            new_axis=0,
            **kwargs,
        )
    else:
        # Non-chunked
        data_pad = np.pad(data, ((radius, radius), (radius, radius)), mode="reflect")
        glcm_cube = block_glcm_textures(data_pad, **kwargs)

    glcm_datacube = xr.DataArray(
        data=glcm_cube,
        dims=["prop", *array_dims],
        coords={
            "prop": [
                "asm",
                "contrast",
                "corr",
                "var",
                "idm",
                "savg",
                "entropy",
                "diffvar",
                "dissimilarity",
            ][:n_feats],
            **array_coords,
        },
        attrs=array_attrs,
    )

    return glcm_datacube


def block_glcm_textures(
    array: np.ndarray,
    radius: int = 1,
    n_feats: int = 7,
    nan_supression: int = 0,
    skip_nan: bool = True,
    rescale_normed: bool = False,
    distances: Union[list, None] = None,
    angles: Union[list, None] = None,
    advanced_idx: bool = False,
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
        - Difference variance
        - Dissimilarity

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
        List of pixel pair angles in radians, by default [0, :math:`\\pi/2`].
    advanced_idx : bool, optional
        Use advanced indexing operations instead of loops when calculating properties that allow it,
        by default False.
    kwargs :
        Other keywords arguments to pass to function
        :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`.

    Returns
    -------
    np.ndarray
        3-dim array with the texture properties for each pixel. The new axis is located at the first
        coordinate, and indexes the properties.

    See Also
    --------
    :func:`matrix <samsara.stats.glcm.matrix>`
    :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`
    :func:`properties <samsara.stats.glcm.properties>`
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

            response[:, i, j] = properties(
                glcm_, n_feats=n_feats, advanced_idx=advanced_idx
            )

    if nan_supression > 0:
        sub_array = array[
            radius : (array.shape[0] - radius), radius : (array.shape[1] - radius)
        ]
        response[
            (np.repeat(sub_array[np.newaxis, :, :], n_feats, axis=0)) == 0
        ] = np.nan

    return response


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
    :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`
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
        glcm_sum = np.sum(glcm_, axis=(0, 1))
        glcm_ = np.divide(
            glcm_, glcm_sum, np.zeros_like(glcm_, dtype=float), where=glcm_sum != 0
        )
    return glcm_


def properties(
    array: np.ndarray,
    n_feats: int = 7,
    summarize: str = "mean",
    advanced_idx: bool = False,
    skip_nan: bool = True,
) -> np.ndarray:
    """Calculate texture features of a gray level co-occurrence matrix.

    From a gray level co-occurrence matrix compute the following properties:

        - ASM
        - Contrast
        - Correlation
        - Variance
        - Inverse Difference Moment
        - Sum Average
        - Entropy
        - Difference variance
        - Dissimilarity

    Parameters
    ----------
    array : np.ndarray
        4-dimensional array. Gray level co-occurrence histogram of an image. The number of times a
        level in the first coordinate occurs at a distance `d` at an angle `t` from another level in
        the second coordinate. The coordinates are (levels, levels, number of distances, number of
        angles).
    n_feats : int, optional
        Number of features or properties computed with the glcm, by default 7.
    summarize: str, optional
        Method to summarize the values of each property, by default 'mean'.
    advanced_idx : bool, optional
        Use advanced indexing operations instead of loops when calculating properties that allow it,
        by default False.
    skip_nan : bool, optional
        When computing the average of each texture for all directions and angles, nan will
        be ignored, by default True.

    Returns
    -------
    np.ndarray
        1-dim array of length `n_feats` with the features or properties computed from `array`.

    Raises
    ------
    ValueError
        If `array` is not 4-dimensional.

    See Also
    --------
    :func:`matrix <samsara.stats.glcm.matrix>`
    :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`
    """
    if array.ndim != 4:
        raise ValueError(f"Matrix should be of 4 dimensions: {array.shape} detected.")

    n_dis = array.shape[2]
    n_ang = array.shape[3]

    # TODO: should be an array of 0s of float32 instead of floeat64?
    ans = np.empty((n_feats, n_dis, n_ang))

    for d in range(n_dis):
        for a in range(n_ang):
            ans[:, d, a] = level_properties(
                array[:, :, d, a], n_feats, advanced_idx=advanced_idx
            )

    if summarize != "mean":
        raise ValueError(
            f"Method to summarize not supported. Expected 'mean', got {summarize}."
        )

    if skip_nan:
        return np.nanmean(ans, axis=(1, 2)).astype(np.float32)
    else:
        return np.mean(ans, axis=(1, 2)).astype(np.float32)


def level_properties(
    array: np.ndarray, n_feats: int = 7, advanced_idx: bool = False
) -> np.ndarray:
    """Calculate texture features of a pair of levels gray level co-occurrence matrix.

    From a slice (of a pair of levels) of the gray level co-occurrence matrix compute the following
    properties:

        - ASM
        - Contrast
        - Correlation
        - Variance
        - Inverse Difference Moment
        - Sum Average
        - Entropy
        - Difference variance
        - Dissimilarity

    Parameters
    ----------
    array : np.ndarray
        2-dimensional array. Occurrence histogram of levels at every distance and angle for one pair
        of levels. The coordinates are (number of distances, number of angles).
    n_feats : int, optional
        Number of features or properties computed, by default 7.
    advanced_idx : bool, optional
        Use advanced indexing operations instead of loops when calculating properties that allow it,
        by default False.

    Returns
    -------
    np.ndarray
        1-dim array of length `n_feats` with the features or properties computed from `array`.

    See Also
    --------
    :func:`properties <samsara.stats.glcm.properties>`
    :func:`matrix <samsara.stats.glcm.matrix>`
    :func:`skimage.feature.graycomatrix <skimage.feature.graycomatrix>`
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

    array_sum = float(np.sum(array))
    p = np.divide(
        array, array_sum, np.zeros_like(array, dtype=float), where=array_sum != 0
    )

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
    px_plus_y, px_minus_y = plus_minus(p, px_plus_y, px_minus_y, advanced_idx)

    fts = [
        np.dot(pravel, pravel),  # 1. ASM
        np.dot(k2, px_minus_y),  # 2. Contrast
        correlation(sx, sy, ux, uy, pravel, ij),  # 3. Correlation
        vx,  # 4. Variance
        np.dot(i_j2_p1, pravel),  # 5. Inverse Difference Moment
        np.dot(tk, px_plus_y),  # 6. Sum Average
        entropy(pravel),  # 9. Entropy
        px_minus_y.var(),  # Difference variance
        np.dot(k, px_minus_y),  # Dissimilarity
    ]

    if len(fts) < n_feats:
        return np.pad(
            np.array(fts, dtype=float), (0, n_feats - len(fts)), constant_values=np.nan
        )

    return fts[:n_feats]
