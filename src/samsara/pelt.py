from typing import Union

import numpy as np
import ruptures as rpt


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


def filter_index_by_date(dates: np.ndarray, start_date: str):
    start_date_np = np.datetime64(start_date)
    indices = np.argwhere(dates > start_date_np).ravel()
    # return np.take(array, indices, axis=0), indices
    return indices


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
    import datetime

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
