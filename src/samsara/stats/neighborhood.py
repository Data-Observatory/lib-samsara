"""Subpackage for neighborhood statistics (`samsara.stats.neighborhood`)
"""
import dask.array as da
import numpy as np
import xarray as xr

__all__ = ["stats", "count", "mean", "sum", "std"]


def count(
    data: xr.Dataset, radius: int = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Count the elements in moving window over an n-dimensional array.

    Get the moving window count over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window count will be calculated.
    radius : int, optional
        Radius of the moving window, by default 0.
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
        If the specified `radius` is larger than any dimension of the array.
    ValueError
        If the specified `radius` is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "count", radius, variable)


def mean(
    data: xr.Dataset, radius: int = 0, variable: str = "magnitude"
) -> xr.DataArray:
    """Calculate moving window mean over an n-dimensional array.

    Get the moving window mean over a Dask array, which is the array named `variable` in the
    dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window mean will be calculated.
    radius : int, optional
        Radius of the moving window, by default 0.
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
        If the specified `radius` is larger than any dimension of the array.
    ValueError
        If the specified `radius` is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "mean", radius, variable)


def sum(data: xr.Dataset, radius: int = 0, variable: str = "magnitude") -> xr.DataArray:
    """Sum the elements in moving window over an n-dimensional array.

    Get the moving window sum over a Dask array, which is the array named `variable` in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window sum will be calculated.
    radius : int, optional
        Radius of the moving window, by default 0.
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
        If the specified `radius` is larger than any dimension of the array.
    ValueError
        If the specified `radius` is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "sum", radius, variable)


def std(data: xr.Dataset, radius: int = 0, variable: str = "magnitude") -> xr.DataArray:
    """Calculate moving window standard deviation over an n-dimensional array.

    Get the moving window standard deviation over a Dask array, which is the array named `variable`
    in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a Dask array over which the moving window standard deviation will be
        calculated.
    radius : int, optional
        Radius of the moving window, by default 0.
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
        If the specified `radius` is larger than any dimension of the array.
    ValueError
        If the specified `radius` is larger than the smallest chunk in any coordinate.

    See Also
    --------
    :func:`stats <samsara.stats.neighborhood.stats>`
    """
    return stats(data, "std", radius, variable)


def stats(
    data: xr.Dataset, stat: str, radius: int = 0, variable: str = "magnitude"
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
    radius : int, optional
        Radius of the moving window, by default 0.
    variable : str, optional
        Data variable of the dataset on which the statistics will be calculated, by default
        'magnitude'.

    Returns
    -------
    xr.DataArray
        Data array with the result of the statistics on the indicated variable.

    Raises
    ------
    ValueError
        If the specified statistic in `stat` is invalid or not supported.
    ValueError
        If the specified `radius` is larger than any dimension of the array.
    ValueError
        If the specified `radius` is larger than the smallest chunk in any coordinate.

    Examples
    --------
    Data creation example:

    >>> import numpy as np
    >>> import xarray as xr
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
    >>> nstat.stats(ds, "count", radius=2, variable="magnitude")
    <xarray.DataArray '_block_stats-94fc541ce6eb03fe87862690338f73fd' (y: 5, x: 4)>
    dask.array<_block_stats, shape=(5, 4), dtype=float64, chunksize=(5, 4), chunktype=numpy.ndarray>
    Coordinates:
    * y        (y) int64 0 1 2 3 4
    * x        (x) int64 0 1 2 3
    """
    if stat not in ["count", "mean", "sum", "std"]:
        raise ValueError(
            "Requested stat not supported. "
            "Currently supported stats are 'count', 'mean', 'sum', 'std'"
        )

    ndim = data[variable].ndim
    signature = str(tuple(i for i in range(ndim))) + "->()"
    wfunc = _get_window_func(stat)
    vfunc = np.vectorize(wfunc, otypes=[float], signature=signature)

    kwargs = {"vfunc": vfunc, "radius": radius}

    # Check size and radius compatibility
    small_side = min(data[variable].data.shape)
    if small_side < radius:
        raise ValueError(
            "Specified radius is larger than your array. The largest window radius for this array"
            f" is {small_side}, that will generate a window of side {small_side * 2 + 1}."
        )

    # Check chunksize and radius compatibility
    small_chunk = min(tuple(j for i in data[variable].data.chunks for j in i))
    if small_chunk < radius:
        raise ValueError(
            "Specified radius is larger than the smallest chunk. The largest window radius for this"
            f" array is {small_chunk}. If you want to use the window radius equal to {radius} you"
            f" must rechunk the data in the variable {variable} so that the size of the smallest"
            f" chunk is equal or larger than {radius}."
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
    )

    return stats_da


def _get_window_func(stat_type: str) -> callable:
    """
    Get the function for the requested stat type
    """
    if stat_type == "count":
        return _window_count
    elif stat_type == "mean":
        return _window_mean
    elif stat_type == "sum":
        return _window_sum
    elif stat_type == "std":
        return _window_std
    else:
        raise ValueError("Invalid stat type.")


def _block_stats(array: np.ndarray, vfunc: callable, radius: int = 0) -> np.ndarray:
    w_side = radius * 2 + 1
    view = np.lib.stride_tricks.sliding_window_view(array, (w_side, w_side))
    return vfunc(view)


def _window_count(array: np.ndarray) -> float:
    center = tuple(i // 2 for i in array.shape)
    if np.isnan(array[center]):
        return np.nan
    return np.count_nonzero(~np.isnan(array))


def _window_sum(array: np.ndarray) -> float:
    center = tuple(i // 2 for i in array.shape)
    if np.isnan(array[center]):
        return np.nan
    return np.nansum(array)


def _window_std(array: np.ndarray) -> float:
    center = tuple(i // 2 for i in array.shape)
    if np.isnan(array[center]):
        return np.nan
    return np.nanstd(array)


def _window_mean(array: np.ndarray) -> float:
    center = tuple(i // 2 for i in array.shape)
    if np.isnan(array[center]):
        return np.nan
    return np.nanmean(array)
