import numpy as np
import pytest
import xarray as xr
from ruptures import KernelCPD, pw_constant

import samsara.pelt as pelt


class TestPelt:
    @pytest.mark.parametrize(
        ("dates", "start_date", "expected"),
        [
            (
                np.array(
                    ["2006-07-13", "2007-01-13", "2010-08-13"], dtype="datetime64"
                ),
                "2006-09-05",
                np.array([1, 2]),
            )
        ],
    )
    def test_filter_index_by_date(self, dates, start_date, expected):
        res = pelt.filter_index_by_date(dates, start_date)
        assert res == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("dates", "expected"),
        [
            (
                np.datetime64("2012-05-12"),
                2012.360656,
            ),
            (
                np.array(
                    ["2007-07-13", "2006-01-13", "2010-08-13"], dtype="datetime64"
                ),
                np.array([2007.528767, 2006.032877, 2010.613699]),
            ),
            (
                np.array(
                    ["2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"],
                    dtype="datetime64",
                ),
                np.array([2020.0, 2020.997267, 2021.0, 2021.997260]),
            ),
            (
                np.array(
                    [
                        "2000-05-12",
                        "2004-05-12",
                        "2008-05-12",
                        "2012-05-12",
                        "2016-05-12",
                        "2020-02-12",
                        "2020-05-12",
                        "2021-02-12",
                        "2021-05-12",
                    ],
                    dtype="datetime64",
                ),
                np.array(
                    [
                        2000.360656,
                        2004.360656,
                        2008.360656,
                        2012.360656,
                        2016.360656,
                        2020.114754,
                        2020.360656,
                        2021.115068,
                        2021.358904,
                    ]
                ),
            ),
        ],
    )
    def test_datetime_to_year_fraction(self, dates, expected):
        res = pelt.datetime_to_year_fraction(dates)
        assert res == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("signal_breaks", "n_breaks", "start_date", "n_nan"),
        [
            (4, 3, None, 0),
            (7, 6, None, 2),
            (7, 6, "2010-05-20", 4),
            (6, 5, "2011-06-30", 5),
            (1, 6, None, 5),
            (1, 6, "2010-05-20", 6),
            (4, 4, None, 0),
        ],
    )
    def test_pixel_pelt(self, signal_breaks, n_breaks, start_date, n_nan):
        array = pw_constant(30, 1, signal_breaks, noise_std=2, seed=723)[0].reshape(30)
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        res = pelt.pixel_pelt(
            array=array,
            dates=dates,
            n_breaks=n_breaks,
            penalty=1,
            start_date=start_date,
        )
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == (n_breaks,)
        assert res[1].shape == (n_breaks,)
        assert np.count_nonzero(np.isnan(res[0])) == n_nan
        assert np.count_nonzero(np.isnan(res[1])) == n_nan

    @pytest.mark.parametrize(
        ("n_breaks", "start_date", "n_nan"),
        [(3, None, 4), (4, None, 14), (3, "2007-03-28", 10)],
    )
    def test_block_pelt(
        self, pelt_signal, pelt_dates, pelt_dates_timestamp, n_breaks, start_date, n_nan
    ):
        array_shape = pelt_signal.shape
        algo_rpt = KernelCPD(kernel="rbf", min_size=3, jump=5)
        res = pelt.block_pelt(
            array=pelt_signal,
            dates=pelt_dates,
            dates_timestamp=pelt_dates_timestamp,
            n_breaks=n_breaks,
            penalty=1,
            start_date=start_date,
            algo_rpt=algo_rpt,
        )
        assert res.shape == (array_shape[1], array_shape[2], 2 * n_breaks)
        assert np.count_nonzero(np.isnan(res)) == n_nan

    def test_block_pelt_int_breakidx(self):
        array = pw_constant(30, 1, 4, noise_std=2, seed=723)[0].reshape((30, 1, 1))
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        dates_timestamp = dates.astype("datetime64[s]").astype(float)
        algo_rpt = KernelCPD(kernel="rbf", min_size=3, jump=5)
        res = pelt.block_pelt(
            array=array,
            dates=dates,
            dates_timestamp=dates_timestamp,
            n_breaks=4,
            penalty=1,
            start_date=None,
            algo_rpt=algo_rpt,
        )
        assert res.shape == (1, 1, 8)
        assert np.count_nonzero(np.isnan(res)) == 0

    @pytest.mark.parametrize(
        ("n_breaks", "start_date"),
        [(3, None), (4, None), (3, "2007-03-28")],
    )
    def test_pelt_xarray(self, pelt_signal_xarray, n_breaks, start_date):
        array_shape = pelt_signal_xarray.shape
        res = pelt.pelt(pelt_signal_xarray, n_breaks, 1, start_date, backend="xarray")
        assert isinstance(res, xr.Dataset)
        assert not {"y", "x", "bkp"} ^ set(res.coords)  # only those 3 values as coords
        assert not {"magnitude", "date"} - set(
            res.variables
        )  # magnitude and date are in vars
        assert res.magnitude.shape == (array_shape[1], array_shape[2], n_breaks)
        assert res.date.shape == (array_shape[1], array_shape[2], n_breaks)

    @pytest.mark.parametrize(
        ("n_breaks", "start_date"),
        [(3, None), (4, None), (3, "2007-03-28")],
    )
    def test_pelt_dask(self, pelt_signal_xarray, n_breaks, start_date):
        array_shape = pelt_signal_xarray.shape
        res = pelt.pelt(pelt_signal_xarray, n_breaks, 1, start_date, backend="dask")
        assert isinstance(res, xr.Dataset)
        assert not {"y", "x", "bkp"} ^ set(res.coords)  # only those 3 values as coords
        assert not {"magnitude", "date"} - set(
            res.variables
        )  # magnitude and date are in vars
        assert res.magnitude.shape == (array_shape[1], array_shape[2], n_breaks)
        assert res.date.shape == (array_shape[1], array_shape[2], n_breaks)

    def test_pelt_model_error(self, pelt_signal_xarray):
        with pytest.raises(
            ValueError, match="Only rbf is accepted as kernel model for KernelCPD"
        ):
            pelt.pelt(pelt_signal_xarray, 5, 1, model="l1")

    def test_pelt_backend_error(self, pelt_signal_xarray):
        with pytest.raises(
            ValueError,
            match="Incorrect backend value. Only 'dask' and 'xarray' are accepted",
        ):
            pelt.pelt(pelt_signal_xarray, 5, 1, backend="other")

    def test_pixel_pelt_empty_array(self):
        array = np.full((30,), np.nan)
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        n_breaks = 5
        res = pelt.pixel_pelt(
            array=array,
            dates=dates,
            n_breaks=n_breaks,
            penalty=30,
            min_size=3,
            jump=5,
        )
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == (n_breaks,)
        assert res[1].shape == (n_breaks,)
        np.testing.assert_array_equal(res[0], np.full(n_breaks, np.nan))
        np.testing.assert_array_equal(res[1], np.full(n_breaks, np.nan))

    def test_pixel_pelt_low_data_array(self):
        # Low amm of data compared to the min_size param, resulting in no possible cofiguration of
        # breakpoints
        array = np.full((30,), np.nan)
        array[-3:] = 10
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        n_breaks = 5
        res = pelt.pixel_pelt(
            array=array,
            dates=dates,
            n_breaks=n_breaks,
            penalty=30,
            min_size=3,
            jump=5,
        )
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == (n_breaks,)
        assert res[1].shape == (n_breaks,)
        np.testing.assert_array_equal(res[0], np.full(n_breaks, np.nan))
        np.testing.assert_array_equal(res[1], np.full(n_breaks, np.nan))

    def test_block_pelt_empty_array(self):
        array = np.full((30, 1, 1), np.nan)
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        dates_timestamp = dates.astype("datetime64[s]").astype(float)
        algo_rpt = KernelCPD(kernel="rbf", min_size=3, jump=5)
        n_breaks = 5
        res = pelt.block_pelt(
            array=array,
            dates=dates,
            dates_timestamp=dates_timestamp,
            n_breaks=n_breaks,
            penalty=30,
            start_date=None,
            algo_rpt=algo_rpt,
        )
        assert res.shape == (1, 1, n_breaks * 2)
        np.testing.assert_array_equal(res, np.full((1, 1, n_breaks * 2), np.nan))

    def test_block_pelt_low_data_array(self):
        array = np.full((30, 1, 1), np.nan)
        array[-3:] = 10
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(30)])
        dates_timestamp = dates.astype("datetime64[s]").astype(float)
        algo_rpt = KernelCPD(kernel="rbf", min_size=3, jump=5)
        n_breaks = 5
        res = pelt.block_pelt(
            array=array,
            dates=dates,
            dates_timestamp=dates_timestamp,
            n_breaks=n_breaks,
            penalty=30,
            start_date=None,
            algo_rpt=algo_rpt,
        )
        assert res.shape == (1, 1, n_breaks * 2)
        np.testing.assert_array_equal(res, np.full((1, 1, n_breaks * 2), np.nan))
