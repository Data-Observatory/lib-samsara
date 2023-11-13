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
) -> xr.DataArray:
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
    xr.DataArray
        3-dim array, the two original positional dimensions and a new one of size equal to twice
        `n_breaks`, where the first `n_breaks` values correspond to the difference of the medians
        between two consecutive breaks, and the following `n_breaks` contain the date on which the
        break occurred.
        If the backend is dask, the new dimension will be in the first position, this means that the
        coordinates will be ('new', 'y', 'x'). If the backend is xarray, the new dimension will be
        in the third position, this means that the coordinates will be ('y', 'x', 'new').

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
    <xarray.DataArray (new: 6, y: 4, x: 5)>
    dask.array<block_pelt, shape=(6, 4, 5), dtype=float64, chunksize=(6, 4, 5)>
    Coordinates:
    * new      (new) int64 0 1 2 3 4 5
    * y        (y) int64 0 1 2 3
    * x        (x) int64 0 1 2 3 4

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
) -> xr.DataArray:
    """
    Apply Pelt using dask and map_blocks.
    """
    data = array.data  # 3d
    dates = array.time.data  # 1d
    chunks = (n_breaks * 2, data.chunks[1], data.chunks[2])
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
        new_axis=0,
    )
    break_xarray = xr.DataArray(
        data=break_cubes,
        dims=["new", array.dims[1], array.dims[2]],
        coords={
            "new": np.arange(n_breaks * 2),
            array.dims[1]: array.coords[array.dims[1]].data,
            array.dims[2]: array.coords[array.dims[2]].data,
        },
    )
    return break_xarray


def pelt_xarray(
    array: xr.DataArray,
    n_breaks: int = 5,
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 1,
) -> xr.DataArray:
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
    break_xarray = xr.apply_ufunc(
        pixel_pelt,
        array,
        input_core_dims=[["time"]],
        output_core_dims=[["new"]],
        exclude_dims={"time"},
        vectorize=True,
        output_dtypes=[array.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {"new": n_breaks * 2},
            "allow_rechunk": True,
        },
        dask="parallelized",
        kwargs=func_kwargs,
    )
    break_xarray = break_xarray.assign_coords({"new": np.arange(n_breaks * 2)})
    return break_xarray
