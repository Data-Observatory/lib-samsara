from typing import Union

import numpy as np

__all__ = ["datetime_to_year_fraction", "filter_index_by_date"]


def filter_index_by_date(dates: np.ndarray, start_date: str) -> np.ndarray:
    """Return indices of dates greater than start_date

    Parameters
    ----------
    dates : np.ndarray
        Array of dates to filter.
    start_date : str
        String with the starting date.

    Returns
    -------
    np.ndarray
        Index array of type int with the indices of the dates grater than the starting date.
    """
    # Make sure that the input array has type datetime
    dates = dates.astype(np.datetime64)
    # Convert start date string to datetime
    start_date_np = np.datetime64(start_date)
    # Filter dates and get indices
    indices = np.where(dates > start_date_np)[0]
    return indices


def datetime_to_year_fraction(dates: np.ndarray) -> Union[np.ndarray, float]:
    """Convert an array of dates to year and fraction.

    Parameters
    ----------
    dates : np.ndarray
        Array of type datetime64 to be converted.

    Returns
    -------
    Union[np.ndarray, float]
        Converted float or array with date as the year and percentage of the year as a decimal.
    """
    year = dates.astype("datetime64[Y]")
    next_year = year + np.timedelta64(1, "Y")
    year_start = year + np.timedelta64(0, "D")
    next_year_start = next_year + np.timedelta64(0, "D")
    # Get year length
    year_length = (next_year_start - year_start) / np.timedelta64(1, "D")
    # Get number of day since the start of the year
    days_elapsed = (dates.astype("datetime64[D]") - year_start) / np.timedelta64(1, "D")
    year_fraction = (year.astype(float) + 1970) + (days_elapsed / year_length)
    return year_fraction
