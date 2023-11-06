from typing import Union

import dask.array as da
import ruptures as rpt
import xarray as xr

from ._block import block_pelt
from ._dates import datetime_to_year_fraction
from ._pixel import pixel_pelt

# from ._pixel import pixel_pelt

__all__ = ["pelt"]


def pelt(
    array: xr.DataArray,
    n_breaks: int = 5,  #
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 5,
    backend: str = "dask",
) -> Union[da.Array, xr.DataArray]:
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
    jump: int = 5,
) -> da.Array:
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

    return break_cubes


def pelt_xarray(
    array: xr.DataArray,
    n_breaks: int = 5,
    penalty: float = 30,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 5,
) -> xr.DataArray:
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
    return break_xarray
