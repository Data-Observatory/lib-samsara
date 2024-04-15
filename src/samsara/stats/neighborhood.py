"""Subpackage for neighborhood statistics (`samsara.stats.neighborhood`)
"""

from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
from numba import njit, prange

from ..kernel import Kernel, square

__all__ = ["stats", "count", "mean", "sum", "std", "quantile"]


def count(
    data: xr.Dataset, kernel: Union[Kernel, int] = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Count the elements in moving window over an n-dimensional array.

    Get the moving window count over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window count will be calculated.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the count is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the count will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the count on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "count", kernel, variable)


def mean(
    data: xr.Dataset, kernel: Union[Kernel, int] = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Calculate moving window mean over an n-dimensional array.

    Get the moving window mean over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window mean will be calculated.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the mean is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the mean will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the mean on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "mean", kernel, variable)


def sum(
    data: xr.Dataset, kernel: Union[Kernel, int] = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Sum the elements in moving window over an n-dimensional array.

    Get the moving window sum over a Dask array, which is the array named `variable` in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window sum will be calculated.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the sum is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the sum will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the sum on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "sum", kernel, variable)


def std(
    data: xr.Dataset, kernel: Union[Kernel, int] = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Calculate moving window standard deviation over an n-dimensional array.

    Get the moving window standard deviation over a Dask array, which is the array named `variable`
    in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window standard deviation will be
        calculated.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the std is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the standard deviation will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the standard deviation on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "std", kernel, variable)


def max(
    data: xr.Dataset, kernel: Union[Kernel, int] = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Calculate moving window maximum over an n-dimensional array.

    Get the moving window maximum over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window maximum will be calculated.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the max is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the maximum will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the maximum on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "max", kernel, variable)


def quantile(
    data: xr.Dataset,
    q: float,
    kernel: Union[Kernel, int] = 0,
    variable: str = "magnitude",
) -> xr.DataArray:
    """Calculate moving window 'q' quantile over an n-dimensional array.

    Get the moving window 'q' quantile over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window maximum will be calculated.
    q : float
        Probability for the quantile to compute. Value must be between 0 and 1 inclusive.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the max is calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the maximum will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array containing the maximum on the indicated variable.

    Raises
    ------
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "quantile", kernel, variable, q)


def stats(
    data: xr.Dataset,
    stat: str,
    kernel: Union[Kernel, int] = 0,
    variable: str = "magnitude",
    q: Union[float, None] = None,
) -> xr.DataArray:
    """Calculate moving window statistics over an n-dimensional array.

    Get the moving window statistics over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window statistic will be calculated.
    stat : str
        Statistic to calculate.
    kernel : Union[Kernel, int], optional
        Kernel used as the moving window and the stats are calculated considering only the valid
        values in it. If the value is an int, a square kernel of radius equal to that value is used,
        by default 0.
    variable : str, optional
        Data variable of the dataset on which the statistics will be calculated, by default
        'magnitude'.
        #TODO ADD Q

    Returns
    -------
    xr.DataArray
        Data array with the result of the statistics on the indicated variable.

    Raises
    ------
    ValueError
        If the specified statistic in `stat` is invalid or not supported.
    ValueError
        If the type of `kernel` is neither Kernel or int.
    ValueError
        If the specified `kernel` radius is larger than any dimension of the array.
    ValueError
        If the specified `kernel` radius is larger than the smallest chunk in any coordinate.

    Examples
    --------
    Data creation example:

    >>> import numpy as np
    >>> import xarray as xr
    >>> import dask.array as da
    >>> mag = da.array(
    ...     [
    ...         [14, 43, 0, 42],
    ...         [28, np.nan, 33, 1],
    ...         [38, np.nan, 20, 18],
    ...         [19, 14, 15, np.nan],
    ...         [np.nan, 46, np.nan, 33],
    ...     ]
    ... )
    >>> ds = xr.Dataset(
    ...     data_vars={"magnitude": (["y", "x"], mag)},
    ...     coords={"y": np.arange(mag.shape[0]), "x": np.arange(mag.shape[1])},
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:    (y: 5, x: 4)
    Coordinates:
    * y          (y) int64 0 1 2 3 4
    * x          (x) int64 0 1 2 3
    Data variables:
        magnitude  (y, x) float64 dask.array<chunksize=(5, 4), meta=np.ndarray>

    Use samsara to get the statistics:

    >>> import samsara.stats.neighborhood as nstat
    >>> nstat.stats(ds, "count", kernel=2, variable="magnitude")
    <xarray.DataArray '_block_stats-94fc541ce6eb03fe87862690338f73fd' (y: 5, x: 4)>
    dask.array<_block_stats, shape=(5, 4), dtype=float64, chunksize=(5, 4), chunktype=numpy.ndarray>
    Coordinates:
    * y        (y) int64 0 1 2 3 4
    * x        (x) int64 0 1 2 3
    """
    if stat not in ["count", "mean", "sum", "std", "max", "quantile"]:
        raise ValueError(
            "Requested stat not supported. "
            "Currently supported stats are 'count', 'mean', 'sum', 'std', 'max', 'quantile'"
        )

    kernel = _check_kernel(kernel)  # Check kernel type
    wfunc = _get_window_func(stat)  # Get function used by stats
    kwargs = {"wfunc": wfunc, "kernel": kernel, "q": q}

    # Kernel radius
    radius = tuple([(i - 1) // 2 for i in kernel.shape])

    # Check size and radius compatibility
    if data[variable].data.shape < radius:
        data_var_shape = data[variable].data.shape
        raise ValueError(
            "Specified kernel radius is larger than your array. The largest kernel shape acceptable"
            f" is {tuple(2 * i + i for i in data_var_shape)}. Otherwise, the largest window radius"
            f" for this array is {min(data_var_shape)}, that will generate a kernel of shape"
            f" {min(data_var_shape) * 2 + 1}."
        )

    # Check chunksize and radius compatibility
    small_chunk = tuple(min(i) for i in data[variable].data.chunks)
    if small_chunk < radius:
        raise ValueError(
            "Specified kernel radius is larger than the smallest chunk. The largest kernel shape"
            f" acceptable for this array is {tuple(2 * i + i for i in small_chunk)}, or a max"
            f" window radius equal to {min(small_chunk)}. If you wanth to use the kernel of shape"
            f" {kernel.shape} you must rechunk the data in the variable {variable} so that the size"
            f" of the smallest chunk is equal or larger than {radius}."
        )

    stats_data = da.map_overlap(
        _block_stats,
        data[variable].data,
        depth=radius,
        boundary=np.nan,
        trim=False,
        dtype=float,
        **kwargs,
    )

    stats_da = xr.DataArray(
        data=stats_data,
        coords=data[variable].coords,
        attrs=data[variable].attrs,
    )

    return stats_da


def _check_kernel(kernel: Union[Kernel, int]) -> Kernel:
    """
    Get the Kernel object for rolling statistics
    """
    if isinstance(kernel, Kernel):
        return kernel
    if isinstance(kernel, int):
        return square(kernel)
    raise ValueError(f"Expected kernel of type Kernel or int, got {type(kernel)}")


def _get_window_func(stat_type: str) -> callable:
    """
    Get the function for the requested stat type
    """
    if stat_type == "count":
        return _view_window_count
    elif stat_type == "mean":
        return _view_window_mean
    elif stat_type == "sum":
        return _view_window_sum
    elif stat_type == "std":
        return _view_window_std
    elif stat_type == "max":
        return _view_window_max
    elif stat_type == "quantile":
        return _view_window_quantile
    else:
        raise ValueError("Invalid stat type.")


def _block_stats(
    array: np.ndarray, wfunc: callable, kernel: Kernel, q: Union[float, None] = None
) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(array, kernel.shape)
    kernel_data = kernel.data
    kernel_data[kernel_data == 0] = np.nan
    if q is not None:
        return wfunc(view, kernel_data, q)
    return wfunc(view, kernel_data)


@njit
def _view_window_count(view: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.count_nonzero(~np.isnan(kernel * subarray))
    return result


@njit
def _view_window_sum(view: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.nansum(kernel * subarray)
    return result


@njit
def _view_window_std(view: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.nanstd(kernel * subarray)
    return result


@njit
def _view_window_mean(view: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.nanmean(kernel * subarray)
    return result


@njit
def _view_window_max(view: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.nanmax(kernel * subarray)
    return result


@njit
def _view_window_quantile(view: np.ndarray, kernel: np.ndarray, q: float) -> np.ndarray:
    result = np.zeros((view.shape[0], view.shape[1]))
    for i in prange(view.shape[0]):
        for j in range(view.shape[1]):
            subarray = view[i, j, :, :]
            if np.isnan(subarray[kernel.shape[0] // 2][kernel.shape[1] // 2]):
                result[i, j] = np.nan
            else:
                result[i, j] = np.nanquantile(kernel * subarray, q)
    return result
