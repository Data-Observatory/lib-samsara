"""
User functions to apply Pelt.
"""
from typing import Union

import dask.array as da
import numpy as np
import ruptures as rpt
import xarray as xr

from ._block import block_pelt
from ._dates import datetime_to_year_fraction
from ._pixel import pixel_pelt

__all__ = ["pelt"]


def pelt(
    array: xr.DataArray,
    n_breaks: int = 5,
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 1,
    backend: str = "dask",
) -> xr.Dataset:
    """Apply the linearly penalized segmentation (Pelt) over a DataArray.

    Apply the Pelt algorithm over every geo-coordinate to find the optimal segmentation in a time
    series.

    Parameters
    ----------
    array : xr.DataArray
        3-dim DataArray, with dimensions ('time', 'y', 'x'), to apply pelt over each (x, y) pair.
    n_breaks : int, optional
        Number of breaks expected in the data, by default 5.
    penalty : float, optional
        Penalty value for the KernelCPD prediction, by default 30.
    start_date : Union[str, None], optional
        Dates from which breaks are calculated, by default None.
    model : str, optional
        Model used by ruptures KernelCPD, by default 'rbf'. Only 'rbf' is supported in the current
        version.
    min_size : int, optional
        Minimum segment length used by ruptures KernelCPD, by default 3.
    jump : int, optional
        Subsample (one every `jump` points), used by ruptures KernelCPD, by default 1.
    backend : str, optional
        Package used to run pelt over the entire array, by default 'dask'. Only 'dask' and 'xarray'
        are supported.

    Returns
    -------
    xr.Dataset
        3-dim array dataset. Contains two arrays, magnitude and date. The magnitude array correspond
        to the difference of the medians between two consecutive breaks. The date array contains the
        date on which the break occurred. Both arrays have the same dimensions, which will be
        ('y', 'x', 'break') if the original array has dimensions ('time', 'y', 'x'). The length of
        the break dimension is equal to `n_breaks`.

    Raises
    ------
    ValueError
        If the value of `model` is other than 'rbf'.
    ValueError
        If the value of `model` is other than 'dask' or 'xarray'.

    Notes
    -----
    The value of `jump` is set to 1 due to ruptures setting not accepting values other than 1 for
    KernelCPD.

    Examples
    --------
    Data creation example:

    >>> import dask.array as da
    >>> import numpy as np
    >>> import xarray as xr
    >>> start_date = np.datetime64('2020-01-01')
    >>> stop_date = np.datetime64('2020-07-01')
    >>> a = xr.DataArray(
    ...     data=da.from_array(np.random.rand(10, 4, 5)),
    ...     dims=["time", "y", "x"],
    ...     coords={
    ...         "time": np.arange(start_date, stop_date, np.timedelta64(20, 'D')).astype("datetime64[ns]"),
    ...         "y":np.arange(4),
    ...         "x":np.arange(5)
    ...     }
    ... )

    Use pelt:

    >>> import samsara.pelt as pelt
    >>> pelt.pelt(a, 3, 1)
    <xarray.Dataset>
    Dimensions:    (y: 4, x: 5, break: 3)
    Coordinates:
    * y          (y) int64 0 1 2 3
    * x          (x) int64 0 1 2 3 4
    * break      (break) int64 0 1 2
    Data variables:
        magnitude  (y, x, break) float64 dask.array<chunksize=(4, 5, 3), meta=np.ndarray>
        date       (y, x, break) float64 dask.array<chunksize=(4, 5, 3), meta=np.ndarray>

    """
    if model != "rbf":
        raise ValueError(
            f"Only rbf is accepted as kernel model for KernelCPD, {model} passed."
        )
    # Choose backend and run pelt
    if backend == "dask":
        return pelt_dask(array, n_breaks, penalty, start_date, model, min_size, jump)
    elif backend == "xarray":
        return pelt_xarray(array, n_breaks, penalty, start_date, model, min_size, jump)
    else:
        raise ValueError(
            f"Incorrect backend value. Only 'dask' and 'xarray' are accepted, {backend} passed"
        )


def pelt_dask(
    array: xr.DataArray,
    n_breaks: int = 5,
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 1,
) -> xr.Dataset:
    """
    Apply Pelt using dask and map_blocks.
    """
    data = array.data  # 3d
    dates = array.time.data  # 1d
    # Recognize coordinates
    notime_dims = [i for i in array.dims if i != "time"]
    coord_0 = notime_dims[0]
    coord_1 = notime_dims[1]
    idx_c0 = array.dims.index(coord_0)
    idx_c1 = array.dims.index(coord_1)
    chunks = (data.chunks[idx_c0], data.chunks[idx_c1], n_breaks * 2)
    # Each chunk, that contains the whole time series, will generate 2 chunks, where the first is
    # the mean magnitude and the second is the dates. There is no problem with iterated magnitude
    # values and dates because the time dimension is not chunked.
    year_fraction = datetime_to_year_fraction(dates)

    algo_rpt = rpt.KernelCPD(kernel=model, min_size=min_size, jump=jump)

    break_cubes = da.map_blocks(
        block_pelt,
        data,
        dates,
        year_fraction,
        n_breaks,
        penalty,
        start_date,
        algo_rpt,
        dtype=float,
        chunks=chunks,
        drop_axis=0,
        new_axis=2,
    )
    magnitude = da.take(break_cubes, np.arange(0, n_breaks), axis=-1)
    date = da.take(break_cubes, np.arange(n_breaks, n_breaks * 2), axis=-1)
    pelt_ds = xr.Dataset(
        data_vars={
            "magnitude": ([coord_0, coord_1, "break"], magnitude),
            "date": ([coord_0, coord_1, "break"], date),
        },
        coords={
            coord_0: array.coords[coord_0].data,
            coord_1: array.coords[coord_1].data,
            "break": np.arange(n_breaks),
        },
    )
    return pelt_ds


def pelt_xarray(
    array: xr.DataArray,
    n_breaks: int = 5,
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 1,
) -> xr.Dataset:
    """
    Apply Pelt using xarray and apply_ufunc.
    """
    func_kwargs = {
        "dates": array.time.data,
        "n_breaks": n_breaks,
        "penalty": penalty,
        "start_date": start_date,
        "model": model,
        "min_size": min_size,
        "jump": jump,
    }
    break_xarrays = xr.apply_ufunc(
        pixel_pelt,
        array,
        input_core_dims=[["time"]],
        output_core_dims=[["break"], ["break"]],
        exclude_dims={"time"},
        vectorize=True,
        output_dtypes=[float, float],
        dask_gufunc_kwargs={
            "output_sizes": {"break": n_breaks},
            "allow_rechunk": True,
        },
        dask="parallelized",
        kwargs=func_kwargs,
    )
    pelt_ds = xr.Dataset(
        data_vars={
            "magnitude": break_xarrays[0],
            "date": break_xarrays[1],
        },
        coords={
            "break": np.arange(n_breaks),
        },
    )

    return pelt_ds
