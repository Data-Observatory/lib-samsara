import dask.array as da
import numpy as np
import xarray as xr

__all__ = ["count", "mean", "sum", "std"]


def count(
    data: xr.Dataset, radius: int = 0, variable: str = "magnitude"
) -> xr.DataArray:
    return stats(data, "count", radius, variable)


def mean(
    data: xr.Dataset, radius: int = 0, variable: str = "magnitude"
) -> xr.DataArray:
    return stats(data, "mean", radius, variable)


def sum(data: xr.Dataset, radius: int = 0, variable: str = "magnitude") -> xr.DataArray:
    return stats(data, "sum", radius, variable)


def std(data: xr.Dataset, radius: int = 0, variable: str = "magnitude") -> xr.DataArray:
    return stats(data, "std", radius, variable)


def stats(
    data: xr.Dataset, stat: str, radius: int = 0, variable: str = "magnitude"
) -> xr.DataArray:
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
