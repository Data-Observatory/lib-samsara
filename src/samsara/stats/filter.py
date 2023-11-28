"""Subpackage to filter values in xarray Dataset (`samsara.stats.filter`)
"""
import numpy as np
import xarray as xr

__all__ = [
    "filter_by_variable",
    "negative_of_first",
    "negative_of_last",
    "first_negative",
    "last_negative",
]


def filter_by_variable(
    data: xr.Dataset, filter_type: str, variable: str = "magnitude"
) -> xr.Dataset:
    """Keep the values that meet a condition on a variable of a Dataset.

    Given a Dataset, for each pixel in every variable/array, find the value that meets the condition
    specified by `filter_type` in the `variable` array. This process reduces the dimensionality of
    each array by 1.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to filter. Must contain two 3-dim arrays, named magnitude and date.
    filter_type : str
        Type of filter to apply. Must be either 'negative_of_first', 'negative_of_last',
        'first_negative', or 'last_negative'.

        - 'negative_of_first'
            Will evaluate that the value in the first index of the 'break' coordinate of the array
            `variable` is between -1 and 0, then it for each array return it values, otherwise the
            returned value is nan.
        - 'negative_of_last'
            Will evaluate that the last value that is not nan in the 'break' coordinate of the array
            `variable` is between -1 and 0, then it for each array return it values, otherwise the
            returned value is nan.
        - 'first_negative'
            Will return the values that are in the index of the first value in the 'break'
            coordinate of the array `variable` that is between -1 and 0. If no value meets this
            criteria, then the returned value is nan.
        - 'last_negative'
            Will return the values that are in the index of the last value in the 'break'
            coordinate of the array `variable` that is between -1 and 0. If no value meets this
            criteria, then the returned value is nan.

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
    """
    func = _get_func(filter_type)
    kwargs = {"variable": variable}
    template = xr.Dataset(
        data_vars={
            "magnitude": data.magnitude.isel({"break": 0}).drop_vars("break"),
            "date": data.date.isel({"break": 0}).drop_vars("break"),
        }
    )
    filter_ds = data.map_blocks(func=func, kwargs=kwargs, template=template)
    return filter_ds


def negative_of_first(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """
    The value of the first break in `variable` must be between -1 and 0 for each pixel
    """
    first = data.isel({"break": 0})
    nof = first.where((first[variable] < 0) & (first[variable] > -1)).drop_vars("break")
    return nof


def negative_of_last(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """
    The value of the last break in `variable` must be between -1 and 0 for each pixel
    """
    magnitude, date = xr.apply_ufunc(
        _pixel_negative_of_last,
        data["magnitude"],
        data["break"],
        input_core_dims=[["break"], ["break"]],
        output_core_dims=[[], []],
        vectorize=True,
        kwargs={"variable": variable},
    )
    nol = xr.Dataset(data_vars={"magnitude": magnitude, "date": date})
    return nol


def first_negative(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """
    The first value where `variable` is between -1 and 0 for each pixel
    """
    magnitude, date = xr.apply_ufunc(
        _pixel_n_negative,
        data["magnitude"],
        data["break"],
        input_core_dims=[["break"], ["break"]],
        output_core_dims=[[], []],
        vectorize=True,
        kwargs={"variable": variable, "n": 0},
    )
    fn = xr.Dataset(data_vars={"magnitude": magnitude, "date": date})
    return fn


def last_negative(data: xr.Dataset, variable: str = "magnitude") -> xr.Dataset:
    """
    The last value where `variable` is between -1 and 0 for each pixel
    """
    magnitude, date = xr.apply_ufunc(
        _pixel_n_negative,
        data["magnitude"],
        data["break"],
        input_core_dims=[["break"], ["break"]],
        output_core_dims=[[], []],
        vectorize=True,
        kwargs={"variable": variable, "n": -1},
    )
    ln = xr.Dataset(data_vars={"magnitude": magnitude, "date": date})
    return ln


def _get_func(filter_type: str) -> callable:
    """
    Get the function that filters the
    """
    if filter_type == "negative_of_first":
        return negative_of_first
    elif filter_type == "negative_of_last":
        return negative_of_last
    elif filter_type == "first_negative":
        return first_negative
    elif filter_type == "last_negative":
        return last_negative
    else:
        raise ValueError("Invalid filter type.")


def _pixel_negative_of_last(
    magnitude: np.ndarray, date: np.ndarray, variable: str = "magnitude"
) -> tuple:
    data = {"magnitude": magnitude, "date": date}
    last_not_nan = np.argwhere(~np.isnan(magnitude))
    idx = last_not_nan[-1] if len(last_not_nan) > 0 else 0  # index of last not nan

    if data[variable][idx] > -1 and data[variable][idx] < 0:
        return magnitude[idx], date[idx]
    else:
        return np.nan, np.nan


def _pixel_n_negative(
    magnitude: np.ndarray, date: np.ndarray, n: int = 0, variable: str = "magnitude"
) -> tuple:
    data = {"magnitude": magnitude, "date": date}
    negatives = np.argwhere((data[variable] < 0) * (data[variable] > -1))

    if len(negatives) == 0:
        return np.nan, np.nan

    idx = negatives[n]
    return magnitude[idx], date[idx]
