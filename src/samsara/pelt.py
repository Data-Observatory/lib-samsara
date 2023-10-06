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
) -> tuple(da.Array, da.Array):
    data = array.data
    dates = array.time.values
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

    return np.vstack(segment_mean_mag, segment_dates)


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
    valid_index: Union[np.ndarray, None],
) -> np.ndarray:
    # Non-jagged output
    algo = rpt.KernelCPD(kernel=model, min_size=3, jump=5)

    def predict_unique_index(array_1d, valid_index):
        breaks = np.full((n_breaks), np.nan)
        breaks_ = algo.fit(array_1d).predict(pen=penalty)
        breaks_ = np.array(breaks_[:-1]) - 1
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
    return


@guvectorize(
    [(float32[:], float32[:], float32[:]), (float64[:], float64[:], float64[:])],
    "(),()->()",
)
def segment_mean(array: np.ndarray, break_idx: np.ndarray):
    return


@guvectorize(
    [(float32[:], float32[:], float32[:]), (float64[:], float64[:], float64[:])],
    "(),()->()",
)
def segment_dates(dates: np.ndarray, break_idx: np.ndarray):
    return


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
