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
def pelt_year_fraction():
    year_fraction = np.array(
        [
            2005.27945,
            2005.43013,
            2005.58082,
            2005.73150,
            2005.88219,
            2006.03287,
            2006.18356,
            2006.33424,
            2006.48493,
            2006.63561,
            2006.78630,
            2006.93698,
            2007.08767,
            2007.23835,
            2007.38904,
            2007.53972,
            2007.69041,
            2007.84109,
            2007.99178,
            2008.14207,
            2008.29234,
            2008.44262,
            2008.59289,
            2008.74316,
            2008.89344,
            2009.04383,
            2009.19452,
            2009.34520,
            2009.49589,
            2009.64657,
            2009.79726,
            2009.94794,
            2010.09863,
            2010.24931,
            2010.40000,
            2010.55068,
            2010.70136,
            2010.85205,
            2011.00273,
            2011.15342,
        ]
    )
    return year_fraction
