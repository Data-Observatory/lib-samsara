from typing import Union

import numpy as np
import ruptures as rpt

from ._dates import datetime_to_year_fraction, filter_index_by_date

__all__ = ["pixel_pelt"]


def pixel_pelt(
    array: np.ndarray,
    dates: np.ndarray,
    n_breaks: int = 5,
    penalty: float = 30.0,
    start_date: Union[str, None] = None,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 5,
) -> np.ndarray:
    # array : dataarray 1dim (time, ) because of input_core_dims
    # dates : len equal to first dim of array of type datetime
    break_idx = pixel_breakpoints_index(array, penalty, model, min_size, jump)

    # Get filtered array and valid indices
    working_idx = None
    if start_date is not None:
        working_idx = filter_index_by_date(dates, start_date)
    # Get valid breakpoint indices
    if working_idx is not None:
        break_idx = np.intersect1d(break_idx, working_idx)
    else:
        break_idx = np.unique(break_idx)

    # Return mean magnitude and segment dates
    if len(break_idx) == 0:
        return np.full((n_breaks), np.nan), np.full((n_breaks), np.nan)

    segment_mean_mag, segment_dates = pixel_segment_metrics(array, dates, break_idx)

    len_segment = len(segment_mean_mag)

    if len_segment < n_breaks:
        segment_mean_mag = np.pad(
            segment_mean_mag.astype(float),
            (0, n_breaks - len_segment),
            mode="constant",
            constant_values=np.nan,
        )
        segment_dates = np.pad(
            segment_dates.astype(float),
            (0, n_breaks - len_segment),
            mode="constant",
            constant_values=np.nan,
        )
        return np.hstack([segment_mean_mag, segment_dates])

    if len_segment > n_breaks:
        return np.hstack([segment_mean_mag[:n_breaks], segment_dates[:n_breaks]])

    return np.hstack([segment_mean_mag, segment_dates])


def pixel_breakpoints_index(
    array: np.ndarray,
    penalty: float,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 5,
) -> list:
    # array: array 1dim
    arr_nnan_idx = np.where(~np.isnan(array))[0]  # Not NaN indices
    arr_nnan = array[arr_nnan_idx]  # Non Nan array
    algo = rpt.KernelCPD(kernel=model, min_size=min_size, jump=jump).fit(arr_nnan)
    breaks_ = algo.predict(pen=penalty)  # Index of breakpoints
    breaks_nnan = np.array(breaks_[:-1], dtype=int) - 1  # Fix breaks to indices values
    breaks = arr_nnan_idx[breaks_nnan]  # Break indices in the original array
    return breaks


def pixel_segment_metrics(array: np.ndarray, dates: np.ndarray, break_idx: np.ndarray):
    break_idx = break_idx.tolist()
    if break_idx[0] != 0:
        break_idx = [0, *break_idx]

    segment_mean = np.zeros(len(break_idx), dtype=float)
    segment_dates = np.zeros(len(break_idx) - 1, dtype=float)
    for i, (j, k) in enumerate(zip(break_idx, break_idx[1:])):
        segment_mean[i] = array[j:k].mean()
        segment_dates[i] = datetime_to_year_fraction(dates[k])

    segment_mean[-1] = array[break_idx[-1] :].mean()

    segment_mean_mag = (
        segment_mean[1:] - segment_mean[:-1]
    )  # length = len(original break_idx)

    return segment_mean_mag, segment_dates