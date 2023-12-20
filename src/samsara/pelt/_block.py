"""
Functions for Pelt applied over blocks/cubes of data.
"""
from typing import Union

import numpy as np
import ruptures as rpt
from numba import float32, float64, guvectorize

from ._dates import filter_index_by_date

__all__ = ["block_pelt"]


def block_pelt(
    array: np.ndarray,
    dates: np.ndarray,
    dates_timestamp: np.ndarray,
    n_breaks: int,
    penalty: float,
    start_date: Union[str, None],
    algo_rpt: rpt.KernelCPD,
) -> np.ndarray:
    """Apply the linearly penalized segmentation (Pelt) over a NumPy Array.

    Apply the Pelt algorithm over every geo-coordinate to find the optimal segmentation in a time
    series.

    Parameters
    ----------
    array : np.ndarray
        3-dim array, with dimensions ('time', 'y', 'x'), to apply pelt over each (x, y) pair.
    dates : np.ndarray
        Array with dates of type np.datetime64.
    dates_timestamp : np.ndarray
        Array with dates of type float. Should be the `dates` array as UNIX time/timestamp.
    n_breaks : int
        Number of breaks expected in the data.
    penalty : float
        Penalty value for the KernelCPD prediction.
    start_date : Union[str, None]
        Dates from which breaks are calculated.
    algo_rpt : rpt.KernelCPD
        KernelCPD instance to use in the search of breakpoints. Must be defined with 'rbf' as model.

    Returns
    -------
    np.ndarray
        3-dim array, the two original positional dimensions and a new one of size equal to twice
        `n_breaks`, where the first `n_breaks` values correspond to the difference of the medians
        between two consecutive breaks, and the following `n_breaks` contain the date on which the
        break occurred. This new dimension will be in the last position, which means that the
        coordinates will be ('y', 'x', 'bkp').
    """
    working_idx = None
    if start_date is not None:
        working_idx = filter_index_by_date(dates, start_date)  # 1d array

    break_idx = block_breakpoints_index(
        array,
        n_breaks,
        penalty,
        algo_rpt,
        valid_index=working_idx,
    )  # 3d array (n_breaks, y, x)

    seg_mean, seg_dates = block_segment_metrics(array, dates_timestamp, break_idx)

    return np.dstack([seg_mean, seg_dates])


def block_breakpoints_index(
    array: np.ndarray,
    n_breaks: int,
    penalty: float,
    algo_rpt: rpt.KernelCPD,
    valid_index: Union[np.ndarray, None],
) -> np.ndarray:
    """
    Get the index of each breakpoint for every pixel/geo-location.
    """

    # Non-jagged output
    def predict_unique_index(array_1d, valid_index):
        algo_rpt.cost.gamma = None
        algo_rpt.cost._gram = None  # pylint: disable=W0212
        breaks = np.full((n_breaks), np.nan)
        # Filter original array to values that are not NaN
        arr_nnan_idx = np.where(~np.isnan(array_1d))[0]  # Not NaN indices
        arr_nnan = array_1d[arr_nnan_idx]  # Non Nan array
        # Check if its possible to have breaks
        if len(arr_nnan) < 1 or not rpt.utils.sanity_check(
            len(arr_nnan), 1, algo_rpt.jump, algo_rpt.min_size
        ):
            return breaks
        breaks_nnan = algo_rpt.fit(arr_nnan).predict(
            pen=penalty
        )  # Predict breaks in not NaN indices
        breaks_nnan = (
            np.array(breaks_nnan[:-1], dtype=int) - 1
        )  # Fix breaks to indices values
        breaks_ = arr_nnan_idx[breaks_nnan]  # Break indices in the original array
        # Get valid breakpoint indices
        if valid_index is not None:
            break_idx = np.intersect1d(breaks_, valid_index)
        else:
            break_idx = np.unique(breaks_)
        # Lowest index that meets the number of expected breaks
        n_breaks_ = min(n_breaks, len(break_idx))
        breaks[:n_breaks_] = break_idx[:n_breaks_]
        return breaks

    breaks = np.apply_along_axis(predict_unique_index, 0, array, valid_index)
    return breaks


def block_segment_metrics(array: np.ndarray, dates: np.ndarray, break_idx: np.ndarray):
    """
    Get the difference of the mean of consecutive segments and the date of each break occurence.
    """
    # dates are year fraction
    array_t = np.transpose(array, axes=(1, 2, 0))  # (time,y,x) -> (y,x,time)
    break_idx_t = np.transpose(break_idx, axes=(1, 2, 0))  # (break,y,x) -> (y,x,break)

    # If break_idx_t is of integer type, it will fail when trying to execute a numba function.
    # Cast to float.
    if break_idx_t.dtype.char in np.typecodes["AllInteger"]:
        break_idx_t = break_idx_t.astype(array_t.dtype)

    seg_mean = np.full_like(break_idx_t, np.nan)
    seg_date = np.full_like(break_idx_t, np.nan)

    segment_metrics(array_t, dates, break_idx_t, seg_mean, seg_date)

    return seg_mean, seg_date


@guvectorize(
    [
        (float32[:], float32[:], float32[:]),
        (float64[:], float64[:], float64[:]),
    ],
    "(times),(bkp)->(bkp)",
    nopython=True,
)
def segment_mean(array, break_idx, seg_mean):
    """
    Get the difference of the mean of consecutive segments.
    """
    if np.isnan(break_idx[0]):
        return

    segment_mean_ = np.zeros(break_idx.shape[0] + 1)

    # First segment value
    idx_f = int(break_idx[0])
    segment_mean_[0] = np.nanmean(array[:idx_f])

    for i in range(break_idx.shape[0] - 1):
        idx_s = int(break_idx[i])  # Segment starting index

        if np.isnan(break_idx[i + 1]):
            segment_mean_[i + 1] = np.nanmean(array[idx_s:])
            seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]
            break

        idx_f = int(break_idx[i + 1])  # Segment final index
        segment_mean_[i + 1] = np.nanmean(array[idx_s:idx_f])

        seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]  # Segment mag

    if np.isnan(break_idx[-1]):
        return

    # The function will reach this statement only if the break_idx array does not have a nan as its
    # last value

    idx_s = int(break_idx[-1])  # Last segment value
    segment_mean_[-1] = np.nanmean(array[idx_s:])

    seg_mean[-1] = segment_mean_[-1] - segment_mean_[-2]  # Last segment mag


@guvectorize(
    [
        (float32[:], float32[:], float32[:]),
        (float64[:], float64[:], float64[:]),
    ],
    "(times),(bkp)->(bkp)",
    nopython=True,
)
def segment_dates(dates, break_idx, seg_date):
    """
    Get the dates of each break occurrence.
    """
    for i in range(break_idx.shape[0]):
        if np.isnan(break_idx[i]):
            return
        idx = int(break_idx[i])
        seg_date[i] = dates[idx]


@guvectorize(
    [
        (float32[:], float32[:], float32[:], float32[:], float32[:]),
        (float64[:], float64[:], float64[:], float64[:], float64[:]),
    ],
    "(times),(times),(bkp)->(bkp),(bkp)",
    nopython=True,
)
def segment_metrics(array, dates, break_idx, seg_mean, seg_date):
    """
    Get the difference of the mean of consecutive segments and the date of each break occurence.
    """
    if np.isnan(break_idx[0]):
        return

    segment_mean_ = np.zeros(break_idx.shape[0] + 1)

    # First segment value
    idx_f = int(break_idx[0])
    segment_mean_[0] = np.nanmean(array[:idx_f])

    for i in range(break_idx.shape[0] - 1):
        idx_s = int(break_idx[i])  # Segment starting index

        seg_date[i] = dates[idx_s]  # Segment date

        if np.isnan(break_idx[i + 1]):
            segment_mean_[i + 1] = np.nanmean(array[idx_s:])
            seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]
            break

        idx_f = int(break_idx[i + 1])  # Segment final index
        segment_mean_[i + 1] = np.nanmean(array[idx_s:idx_f])

        seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]  # Segment mag

    if np.isnan(break_idx[-1]):
        return

    # The function will reach this statement only if the break_idx array does not have a nan as its
    # last value

    idx_s = int(break_idx[-1])  # Last segment value
    segment_mean_[-1] = np.nanmean(array[idx_s:])

    seg_mean[-1] = segment_mean_[-1] - segment_mean_[-2]  # Last segment mag
    seg_date[-1] = dates[idx_s]  # Last segment date
