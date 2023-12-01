import dask.array as da
import numpy as np
import pytest
import xarray as xr
from ruptures import pw_constant


@pytest.fixture
def pelt_signal():
    signal = np.full((40, 2, 3), np.nan)
    for i in range(3):
        sig_, _ = pw_constant(
            signal.shape[0], signal.shape[1], i + 3, noise_std=2, seed=16
        )
        signal[:, :, i] = sig_
        signal[:5, :, i] = np.nan
        signal[10, :, i] = np.nan
        signal[20:25, :, i] = np.nan
    return signal


@pytest.fixture
def pelt_dates():
    dates = np.array(
        [
            np.datetime64("2005-04-13") + np.timedelta64(str(55 * i), "D")
            for i in range(40)
        ]
    )
    return dates


@pytest.fixture
def pelt_signal_dask(pelt_signal):
    signal_da = da.from_array(pelt_signal, chunks=(40, 1, 2))
    return signal_da


@pytest.fixture
def pelt_signal_xarray(pelt_signal_dask, pelt_dates):
    signal_xr = xr.DataArray(
        data=pelt_signal_dask,
        dims=["time", "y", "x"],
        coords={
            "time": pelt_dates.astype("datetime64[ns]"),
            "y": np.arange(2),
            "x": np.arange(3),
        },
    )
    return signal_xr


@pytest.fixture
def pelt_dates_timestamp():
    year_fraction = np.array(
        [
            1113350400,
            1118102400,
            1122854400,
            1127606400,
            1132358400,
            1137110400,
            1141862400,
            1146614400,
            1151366400,
            1156118400,
            1160870400,
            1165622400,
            1170374400,
            1175126400,
            1179878400,
            1184630400,
            1189382400,
            1194134400,
            1198886400,
            1203638400,
            1208390400,
            1213142400,
            1217894400,
            1222646400,
            1227398400,
            1232150400,
            1236902400,
            1241654400,
            1246406400,
            1251158400,
            1255910400,
            1260662400,
            1265414400,
            1270166400,
            1274918400,
            1279670400,
            1284422400,
            1289174400,
            1293926400,
            1298678400,
        ]
    ).astype(float)
    return year_fraction
