import dask.array as da
import numpy as np
import pytest
import xarray as xr

import samsara.stats.neighborhood as ssn


class TestStatsNeighborhood:
    @pytest.fixture
    def data(self):
        mag = da.array(
            [
                [14, 43, 0, 42],
                [28, np.nan, 33, 1],
                [38, np.nan, 20, 18],
                [19, 14, 15, np.nan],
                [np.nan, 46, np.nan, 33],
            ]
        )
        ds = xr.Dataset(
            data_vars={"magnitude": (["y", "x"], mag)},
            coords={"y": np.arange(mag.shape[0]), "x": np.arange(mag.shape[1])},
        )
        return ds

    def test_stats_error(self, data):
        with pytest.raises(ValueError, match="Requested stat not supported."):
            ssn.stats(data, "half", kernel=1, variable="magnitude")

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                2,
                np.array(
                    [
                        [7, 10, 10, 7],
                        [10, np.nan, 13, 9],
                        [11, np.nan, 15, 11],
                        [8, 11, 11, np.nan],
                        [np.nan, 8, np.nan, 6],
                    ]
                ),
            ),
            (
                3,
                np.array(
                    [
                        [13, 13, 13, 13],
                        [15, np.nan, 15, 15],
                        [15, np.nan, 15, 15],
                        [15, 15, 15, np.nan],
                        [np.nan, 11, np.nan, 11],
                    ]
                ),
            ),
            (
                4,
                np.array(
                    [
                        [15, 15, 15, 15],
                        [15, np.nan, 15, 15],
                        [15, np.nan, 15, 15],
                        [15, 15, 15, np.nan],
                        [np.nan, 15, np.nan, 15],
                    ]
                ),
            ),
        ],
    )
    def test_stats_window_bigger_than_chunk_1c(self, data, kernel, expected):
        # Window is bigger than the chunk
        res = ssn.stats(data, "count", kernel=kernel)
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                2,
                np.array(
                    [
                        [7, 10, 10, 7],
                        [10, np.nan, 13, 9],
                        [11, np.nan, 15, 11],
                        [8, 11, 11, np.nan],
                        [np.nan, 8, np.nan, 6],
                    ]
                ),
            ),
        ],
    )
    def test_stats_chunks(self, data, kernel, expected):
        # Window is bigger than the chunk
        data.magnitude.data = data.magnitude.data.rechunk((3, 2))
        res = ssn.stats(data, "count", kernel=kernel)
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)

    @pytest.mark.parametrize(
        ("kernel"),
        [5, 6, 7],
    )
    def test_stats_error_kernel_bigger_than_shape(self, data, kernel):
        with pytest.raises(
            ValueError, match="Specified kernel radius is larger than your array"
        ):
            ssn.stats(data, "count", kernel=kernel)

    @pytest.mark.parametrize(
        ("kernel", "chunks"),
        [(3, (3, 2)), (4, (3, 2)), (2, (2, 2))],
    )
    def test_stats_error_kernel_bigger_than_chunk(self, data, kernel, chunks):
        # Window is bigger than the chunk
        data.magnitude.data = data.magnitude.data.rechunk(chunks)
        with pytest.raises(
            ValueError,
            match="Specified kernel radius is larger than the smallest chunk",
        ):
            ssn.stats(data, "count", kernel=kernel)

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                0,
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, np.nan, 0, 0],
                        [0, np.nan, 0, 0],
                        [0, 0, 0, np.nan],
                        [np.nan, 0, np.nan, 0],
                    ]
                ),
            ),
            (
                1,
                np.array(
                    [
                        [11.84154645, 15.05456741, 19.34321587, 18.77498336],
                        [11.07643896, np.nan, 16.49984539, 15.340578],
                        [9.14808723, np.nan, 9.44133936, 10.24890238],
                        [13.17905535, 12.18833687, 11.52774431, np.nan],
                        [np.nan, 13.12440475, np.nan, 9.0],
                    ]
                ),
            ),
        ],
    )
    def test_std(self, data, kernel, expected):
        res = ssn.std(data, kernel, variable="magnitude")
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                0,
                np.array(
                    [
                        [1, 1, 1, 1],
                        [1, np.nan, 1, 1],
                        [1, np.nan, 1, 1],
                        [1, 1, 1, np.nan],
                        [np.nan, 1, np.nan, 1],
                    ]
                ),
            ),
            (
                1,
                np.array(
                    [
                        [3, 5, 5, 4],
                        [4, np.nan, 7, 6],
                        [4, np.nan, 6, 5],
                        [4, 6, 6, np.nan],
                        [np.nan, 4, np.nan, 2],
                    ]
                ),
            ),
        ],
    )
    def test_count(self, data, kernel, expected):
        res = ssn.count(data, kernel, variable="magnitude")
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                0,
                np.array(
                    [
                        [14, 43, 0, 42],
                        [28, np.nan, 33, 1],
                        [38, np.nan, 20, 18],
                        [19, 14, 15, np.nan],
                        [np.nan, 46, np.nan, 33],
                    ]
                ),
            ),
            (
                1,
                np.array(
                    [
                        [28.333333, 23.6, 23.8, 19.0],
                        [30.75, np.nan, 22.428572, 19.0],
                        [24.75, np.nan, 16.833333, 17.4],
                        [29.25, 25.333333, 24.333333, np.nan],
                        [np.nan, 23.5, np.nan, 24.0],
                    ]
                ),
            ),
        ],
    )
    def test_mean(self, data, kernel, expected):
        res = ssn.mean(data, kernel, variable="magnitude")
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)

    @pytest.mark.parametrize(
        ("kernel", "expected"),
        [
            (
                0,
                np.array(
                    [
                        [14, 43, 0, 42],
                        [28, np.nan, 33, 1],
                        [38, np.nan, 20, 18],
                        [19, 14, 15, np.nan],
                        [np.nan, 46, np.nan, 33],
                    ]
                ),
            ),
            (
                1,
                np.array(
                    [
                        [85.0, 118.0, 119.0, 76.0],
                        [123.0, np.nan, 157.0, 114.0],
                        [99.0, np.nan, 101.0, 87.0],
                        [117.0, 152.0, 146.0, np.nan],
                        [np.nan, 94.0, np.nan, 48.0],
                    ]
                ),
            ),
        ],
    )
    def test_sum(self, data, kernel, expected):
        res = ssn.sum(data, kernel, variable="magnitude")
        assert res.shape == data.magnitude.shape
        res_data = res.data.compute()
        np.testing.assert_array_almost_equal(res_data, expected)
