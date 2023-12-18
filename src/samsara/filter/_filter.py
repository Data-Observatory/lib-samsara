"""
User functions to apply filters.
"""
import numpy as np
import xarray as xr

__all__ = [
    "filter_by_variable",
    "negative_of",
    "negative_of_last",
    "first_negative",
    "last_negative",
]


def filter_by_variable(
    data: xr.Dataset, filter_type: str, bkp_index: int = 0, variable: str = "magnitude"
) -> xr.Dataset:
    """Keep the values that meet a condition on a variable of a Dataset.

    Given a Dataset, for each pixel in every variable/array, find the value that meets the condition
    specified by `filter_type` in the `variable` array. This process reduces the dimensionality of
    each array by 1.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to filter. Must contain two 3-dim Dask arrays, named magnitude and date. The third
        dimension/coordinate will be reduced, and must be named 'bkp'.
    filter_type : str
        Type of filter to apply. Must be either 'negative_of', 'negative_of_last',
        'first_negative', or 'last_negative'.

        - 'negative_of'
            Will evaluate that the value in the `bkp_index` index of the 'bkp' coordinate of the array
            `variable` is between -1 and 0, then it for each array return it values, otherwise the
            returned value is nan.
        - 'negative_of_last'
            Will evaluate that the last value that is not nan in the 'bkp' coordinate of the array
            `variable` is between -1 and 0, then it for each array return it values, otherwise the
            returned value is nan.
        - 'first_negative'
            Will return the values that are in the index of the first value in the 'bkp'
            coordinate of the array `variable` that is between -1 and 0. If no value meets this
            criteria, then the returned value is nan.
        - 'last_negative'
            Will return the values that are in the index of the last value in the 'bkp'
            coordinate of the array `variable` that is between -1 and 0. If no value meets this
            criteria, then the returned value is nan.
    bkp_index : int, optional
        Used in 'negative_of'. The index of the bkp coordinate from which the values will be
        obtained only if they meet the negativity condition, by default 0.
    variable : str, optional
        Name of the array on which the conditions will be evaluated, by default 'magnitude'.

    Returns
    -------
    xr.Dataset
        2-dim array dataset. Contains two arrays, magnitude and date. Each cell of the arrays
        contains the value that meets the condition of the selected filter.

    Raises
    ------
    ValueError
        If the filter type is not supported. Currently supported types are 'negative_of_first',
        'negative_of_last', 'first_negative', 'last_negative'.

    Examples
    --------

    Data creation example:

    >>> import dask.array as da
    >>> import numpy as np
    >>> import xarray as xr
    >>> mag = da.array(
    ...     [
    ...         [
    ...             [np.nan, np.nan, np.nan],
    ...             [0.0292, -0.3283, np.nan],
    ...             [0.3207, -0.8798, -0.9838],
    ...             [0.4581, np.nan, np.nan],
    ...         ],
    ...         [
    ...             [0.1838, -0.3835, np.nan],
    ...             [-0.4497, 0.9151, np.nan],
    ...             [0.1864, -0.1234, 0.5554],
    ...             [-0.0617, -0.8852, 0.0588],
    ...         ],
    ...     ]
    ... )
    >>> dat = da.array(
    ...     [
    ...         [
    ...             [np.nan, np.nan, np.nan],
    ...             [1107475200, 1107561600, np.nan],
    ...             [1107734400, 1107820800, 1107907200],
    ...             [1107993600, np.nan, np.nan],
    ...         ],
    ...         [
    ...             [1108252800, 1108339200, np.nan],
    ...             [1108512000, 1108598400, np.nan],
    ...             [1108771200, 1108857600, 1108944000],
    ...             [1109030400, 1109116800, 1109203200],
    ...         ],
    ...     ]
    ... )
    >>> y, x, brk = mag.shape
    >>> ds = xr.Dataset(
    ...     data_vars={
    ...         "magnitude": (["y", "x", "bkp"], mag),
    ...         "date": (["y", "x", "bkp"], dat),
    ...     },
    ...     coords={
    ...         "y": np.arange(y),
    ...         "x": np.arange(x),
    ...         "bkp": np.arange(brk),
    ...     },
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:    (y: 2, x: 4, bkp: 3)
    Coordinates:
    * y          (y) int64 0 1
    * x          (x) int64 0 1 2 3
    * bkp      (bkp) int64 0 1 2
    Data variables:
        magnitude  (y, x, bkp) float64 dask.array<chunksize=(2, 4, 3), meta=np.ndarray>
        date       (y, x, bkp) float64 dask.array<chunksize=(2, 4, 3), meta=np.ndarray>

    Use samsara to filter the dataset:

    >>> import samsara.filter as sfilter
    >>> sfilter.filter_by_variable(ds, "negative_of_last", variable="magnitude")
    <xarray.Dataset>
    Dimensions:    (y: 2, x: 4)
    Coordinates:
    * y          (y) int64 0 1
    * x          (x) int64 0 1 2 3
    Data variables:
        magnitude  (y, x) float64 dask.array<chunksize=(2, 4), meta=np.ndarray>
        date       (y, x) float64 dask.array<chunksize=(2, 4), meta=np.ndarray>

    """
    func = _get_func(filter_type)

    other_vars = list(set(data.data_vars.keys()) - set({variable}))
    # For now, there is only support for one other variable apart from the main one
    variable_1 = other_vars[0]

    kwargs = {"variable": variable}

    # Add bkp_index to kwargs only if it's used
    if filter_type == "negative_of":
        kwargs["bkp_index"] = bkp_index

    template = xr.Dataset(
        data_vars={
            variable: data[variable].isel({"bkp": 0}).drop_vars("bkp"),
            variable_1: data[variable_1].isel({"bkp": 0}).drop_vars("bkp"),
        }
    )
    filter_ds = data.map_blocks(func=func, kwargs=kwargs, template=template)
    return filter_ds


def negative_of(
    data: xr.Dataset, bkp_index: int = 0, variable: str = "magnitude"
) -> xr.Dataset:
    """Filter an in-memory dataset keeping the negatives of the `bkp_index` index.

    The value of the `bkp_index`-th break in `variable` must be between -1 and 0 for each pixel.
    """
    if bkp_index >= len(data["bkp"]):
        raise IndexError(
            f"Invalid bkp_index. Got index {bkp_index} for coordinate of length {len(data['bkp'])}"
        )
    of = data.isel({"bkp": bkp_index})
    neg_of = of.where((of[variable] < 0) & (of[variable] > -1)).drop_vars("bkp")
    return neg_of


def negative_of_last(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """Filter an in-memory dataset keeping the negatives of the last index.

    The value of the last break in `variable` must be between -1 and 0 for each pixel.
    """
    other_vars = list(set(data.data_vars.keys()) - set({variable}))
    # For now, there is only support for one other variable apart from the main one
    variable_1 = other_vars[0]
    filter_xarray = xr.apply_ufunc(
        _pixel_negative_of_last,
        data[variable],
        data[variable_1],
        input_core_dims=[["bkp"], ["bkp"]],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True,
    )
    nol = xr.Dataset(
        data_vars={variable: filter_xarray[0], variable_1: filter_xarray[1]}
    )
    return nol


def first_negative(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """Filter an in-memory dataset keeping the first negative.

    The first value where `variable` is between -1 and 0 for each pixel.
    """
    other_vars = list(set(data.data_vars.keys()) - set({variable}))
    # For now, there is only support for one other variable apart from the main one
    variable_1 = other_vars[0]

    filter_xarray = xr.apply_ufunc(
        _pixel_n_negative,
        data[variable],
        data[variable_1],
        input_core_dims=[["bkp"], ["bkp"]],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True,
        kwargs={"n": 0},
    )
    fn = xr.Dataset(
        data_vars={variable: filter_xarray[0], variable_1: filter_xarray[1]}
    )
    return fn


def last_negative(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """Filter an in-memory dataset keeping the last negative.

    The last value where `variable` is between -1 and 0 for each pixel.
    """
    other_vars = list(set(data.data_vars.keys()) - set({variable}))
    # For now, there is only support for one other variable apart from the main one
    variable_1 = other_vars[0]

    filter_xarray = xr.apply_ufunc(
        _pixel_n_negative,
        data[variable],
        data[variable_1],
        input_core_dims=[["bkp"], ["bkp"]],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True,
        kwargs={"n": -1},
    )
    ln = xr.Dataset(
        data_vars={variable: filter_xarray[0], variable_1: filter_xarray[1]}
    )
    return ln


def _get_func(filter_type: str) -> callable:
    """
    Get the function for the requested filter type
    """
    if filter_type == "negative_of":
        return negative_of
    elif filter_type == "negative_of_last":
        return negative_of_last
    elif filter_type == "first_negative":
        return first_negative
    elif filter_type == "last_negative":
        return last_negative
    else:
        raise ValueError("Invalid filter type.")


def _pixel_negative_of_last(data: np.ndarray, date: np.ndarray) -> tuple:
    last_not_nan = np.nonzero(~np.isnan(data))[0]
    idx = last_not_nan[-1] if len(last_not_nan) > 0 else 0  # index of last not nan

    if data[idx] > -1 and data[idx] < 0:
        return data[idx], date[idx]
    else:
        return np.nan, np.nan


def _pixel_n_negative(data: np.ndarray, date: np.ndarray, n: int = 0) -> tuple:
    negatives = np.nonzero((data < 0) * (data > -1))[0]

    if len(negatives) == 0:
        return np.nan, np.nan

    idx = negatives[n]
    return data[idx], date[idx]
