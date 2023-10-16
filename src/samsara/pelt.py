import datetime
from typing import Union

import dask.array as da
import numpy as np
import ruptures as rpt
import xarray as xr
from numba import float32, float64, guvectorize

# For the whole image in the whole time series


def pelt(
    array: xr.DataArray,
    n_breaks: int = 5,  #
    penalty: float = 30,
    start_date: Union[str, None] = None,
    **kwargs,
) -> tuple[da.Array, da.Array]:
    data = array.data
    dates = array.time.values  # 1d
    chunks = ((n_breaks, n_breaks), data.chunks[1], data.chunks[2])
    # Each chunk, that contains the whole time series, will generate 2 chunks, where the first is
    # the mean magnitude and the second is the dates. There is no problem with iterated magnitude
    # values and dates because the time dimension is not chunked.

    break_cubes = da.map_blocks(
        block_pelt,
        data,
        dates,
        n_breaks,
        penalty,
        start_date,
        **kwargs,
        dtype=float,
        chunks=chunks,
        drop_axis=0,
        new_axis=0,
    )

    magnitude_cube = break_cubes[:n_breaks]
    dates_cube = break_cubes[n_breaks:]

    return magnitude_cube, dates_cube


def datetime_to_year_fraction(dates):
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


def block_pelt(
    array: np.ndarray,
    dates: np.ndarray,
    n_breaks: int,
    penalty: float,
    start_date: Union[str, None],
    kernel_model: str = "rbf",
) -> np.ndarray:
    working_idx = None
    if start_date is None:
        working_idx = filter_index_by_date(dates, start_date)  # 1d array

    break_idx = block_breakpoints_index(
        array, penalty, n_breaks, model=kernel_model, valid_index=working_idx
    )  # 3d array (n_breaks, y, x)

    segment_mean_mag, segment_dates = block_segment_metrics(array, dates, break_idx)

    return np.vstack([segment_mean_mag, segment_dates])


def filter_index_by_date(dates: np.ndarray, start_date: str):
    start_date_np = np.datetime64(start_date)
    indices = np.argwhere(dates > start_date_np).ravel()
    # return np.take(array, indices, axis=0), indices
    return indices


def block_breakpoints_index(
    array: np.ndarray,
    penalty: float,
    n_breaks: int,
    model: str,
    min_size: int,
    jump: int,
    valid_index: Union[np.ndarray, None],
) -> np.ndarray:
    # Non-jagged output

    if model != "rbf":
        raise ValueError(
            f"Only rbf is accepted as kernel model for KernelCPD, {model} passed."
        )

    algo = rpt.KernelCPD(kernel=model, min_size=min_size, jump=jump)

    def predict_unique_index(array_1d, valid_index):
        algo.cost.gamma = None
        algo.cost._gram = None
        breaks = np.full((n_breaks), np.nan)
        # Filter original array to values that are not NaN
        arr_nnan_idx = np.where(~np.isnan(array_1d))[0]  # Not NaN indices
        arr_nnan = array_1d[arr_nnan_idx]  # Non Nan array
        breaks_nnan = algo.fit(arr_nnan).predict(
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
    # dates are year fraction
    array_t = np.transpose(array, axes=(1, 2, 0))  # (time,y,x) -> (y,x,time)
    break_idx_t = np.transpose(break_idx, axes=(1, 2, 0))  # (break,y,x) -> (y,x,break)

    # If break_idx_t is of integer type, it will fail when trying to execute a numba function.
    # Cast to float.
    if break_idx_t.dtype.char in np.typecodes["AllInteger"]:
        break_idx_t = break_idx_t.astype(array_t.dtype)

    seg_mean = np.full_like(break_idx_t, np.nan)
    seg_date = np.full_like(break_idx_t, np.nan)

    segment_mean(array_t, break_idx_t, seg_mean)
    segment_dates(dates, break_idx_t, seg_date)

    seg_mean = np.transpose(seg_mean, axes=(2, 0, 1))
    seg_date = np.transpose(seg_date, axes=(2, 0, 1))

    return seg_mean, seg_date


@guvectorize(
    [
        (float32[:], float32[:], float32[:]),
        (float64[:], float64[:], float64[:]),
    ],
    "(times),(break)->(break)",
    nopython=True,
)
def segment_mean(array, break_idx, seg_mean):
    if np.isnan(break_idx[0]):
        return

    segment_mean_ = np.zeros(break_idx.shape[0] + 1)

    # First segment value
    idx_f = int(break_idx[0])
    segment_mean_[0] = array[:idx_f].mean()

    for i in range(break_idx.shape[0] - 1):
        idx_s = int(break_idx[i])  # Segment starting index

        if np.isnan(break_idx[i + 1]):
            segment_mean_[i + 1] = array[idx_s:].mean()
            seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]
            break

        idx_f = int(break_idx[i + 1])  # Segment final index
        segment_mean_[i + 1] = array[idx_s:idx_f].mean()

        seg_mean[i] = segment_mean_[i + 1] - segment_mean_[i]  # Segment mag

    if np.isnan(break_idx[-1]):
        return

    # The function will reach this statement only if the break_idx array does not have a nan as its
    # last value

    idx_s = int(break_idx[-1])  # Last segment value
    segment_mean_[-1] = array[idx_s:].mean()

    seg_mean[-1] = segment_mean_[-1] - segment_mean_[-2]  # Last segment mag


@guvectorize(
    [
        (float32[:], float32[:], float32[:]),
        (float64[:], float64[:], float64[:]),
    ],
    "(times),(break)->(break)",
    nopython=True,
)
def segment_dates(dates, break_idx, seg_date):
    for i in range(break_idx.shape[0]):
        if np.isnan(break_idx[i]):
            return
        idx = int(break_idx[i])
        seg_date[i] = dates[idx]


# For 1 pixel


def breakpoints(
    array: np.ndarray,
    dates: np.ndarray,
    start_date: Union[str, None] = None,
    penalty: float = 30.0,
) -> np.ndarray:
    # array : dataarray 1dim (time, ) because of input_core_dims
    # dates : len equal to first dim of array of type datetime
    break_idx = breakpoints_index(array, penalty)

    # Get filtered array and valid indices
    working_idx = None
    if start_date is not None:
        working_idx = filter_index_by_date(dates, start_date)  # 3 dim

    # Get valid breakpoint indices
    if working_idx is not None:
        break_idx = np.intersect1d(break_idx, working_idx)
    else:
        break_idx = np.unique(break_idx)

    # Return mean magnitude and segment dates
    if len(break_idx) == 0:
        return np.array([]), np.array([])

    segment_mean_mag, segment_dates = segment_metrics(array, dates, break_idx)
    return segment_mean_mag, segment_dates


def breakpoints_index(array: np.ndarray, penalty: float, model: str = "rbf") -> list:
    # array: array 1dim
    algo = rpt.KernelCPD(kernel=model, min_size=3, jump=5).fit(array)
    breaks = algo.predict(pen=penalty)  # Index of breakpoints
    breaks = breaks[:-1]
    breaks = [n - 1 for n in breaks]
    return breaks


def segment_metrics(array: np.ndarray, dates: np.ndarray, break_idx: np.ndarray):
    break_idx = break_idx.tolist()
    if break_idx[0] != 0:
        break_idx = [0, *break_idx]

    segment_mean = np.zeros(len(break_idx), dtype=float)
    segment_dates = np.zeros(len(break_idx) - 1, dtype=float)
    for i, (j, k) in enumerate(zip(break_idx, break_idx[1:])):
        segment_mean[i] = array[j:k].mean()
        segment_dates[i] = _year_fraction(dates[k])

    segment_mean[-1] = array[break_idx[-1] :].mean()

    segment_mean_mag = (
        segment_mean[1:] - segment_mean[:-1]
    )  # length = len(original break_idx)

    return segment_mean_mag, segment_dates


def _year_fraction(date):
    year = date.astype("datetime64[Y]").astype(int) + 1970
    month = date.astype("datetime64[M]").astype(int) % 12 + 1
    day = (date - date.astype("datetime64[M]") + 1).astype(int)
    year_start = datetime.date(year, 1, 1).toordinal()
    year_length = datetime.date(year + 1, 1, 1).toordinal() - year_start
    return (
        year
        + float(datetime.date(year, month, day).toordinal() - year_start) / year_length
    )


# Legacy
# ------
# Create dataframe  with ix equal to enumerate data
# The values of index is equal to
#
# def create_df(data, dates):
#     df = pd.DataFrame(index=pd.to_datetime(dates),
#                             data={'ndvi':data.ravel(), 'ix':range(len(data))})
#     df = df.dropna(how='any')
#     return df
