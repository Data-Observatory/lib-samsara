import dask.array as da
import numpy as np
import pytest
import xarray as xr

import samsara.filter as sfilter


class TestFilterVariable:
    @pytest.fixture
    def dataset(self):
        mag = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [0.0292, -0.3283, np.nan],
                    [0.3207, -0.8798, -0.9838],
                    [0.4581, np.nan, np.nan],
                ],
                [
                    [0.1838, -0.3835, np.nan],
                    [-0.4497, 0.9151, np.nan],
                    [0.1864, -0.1234, 0.5554],
                    [-0.0617, -0.8852, 0.0588],
                ],
            ]
        )
        dat = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [1107475200, 1107561600, np.nan],
                    [1107734400, 1107820800, 1107907200],
                    [1107993600, np.nan, np.nan],
                ],
                [
                    [1108252800, 1108339200, np.nan],
                    [1108512000, 1108598400, np.nan],
                    [1108771200, 1108857600, 1108944000],
                    [1109030400, 1109116800, 1109203200],
                ],
            ]
        )
        y, x, brk = mag.shape
        ds = xr.Dataset(
            data_vars={
                "magnitude": (["y", "x", "break"], mag),
                "date": (["y", "x", "break"], dat),
            },
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "break": np.arange(brk),
            },
        )
        return ds

    @pytest.mark.parametrize(
        ("variable", "expected"),
        [
            (
                "magnitude",
                (
                    np.array(
                        [
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, -0.4497, np.nan, -0.0617],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, 1108512000, np.nan, 1109030400],
                        ]
                    ),
                ),
            )
        ],
    )
    def test_negative_of_first(self, dataset, variable, expected):
        ds = sfilter.negative_of_first(dataset, variable=variable)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        np.testing.assert_array_almost_equal(ds.magnitude.data, expected[0])
        np.testing.assert_array_almost_equal(ds.date.data, expected[1])

    @pytest.mark.parametrize(
        ("variable", "expected"),
        [
            (
                "magnitude",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.9838, np.nan],
                            [-0.3835, np.nan, np.nan, np.nan],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107907200, np.nan],
                            [1108339200, np.nan, np.nan, np.nan],
                        ]
                    ),
                ),
            )
        ],
    )
    def test_negative_of_last(self, dataset, variable, expected):
        ds = sfilter.negative_of_last(dataset, variable=variable)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        np.testing.assert_array_almost_equal(ds.magnitude.data, expected[0])
        np.testing.assert_array_almost_equal(ds.date.data, expected[1])

    @pytest.mark.parametrize(
        ("variable", "expected"),
        [
            (
                "magnitude",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.8798, np.nan],
                            [-0.3835, -0.4497, -0.1234, -0.0617],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107820800, np.nan],
                            [1108339200, 1108512000, 1108857600, 1109030400],
                        ]
                    ),
                ),
            )
        ],
    )
    def test_first_negative(self, dataset, variable, expected):
        ds = sfilter.first_negative(dataset, variable=variable)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        np.testing.assert_array_almost_equal(ds.magnitude.data, expected[0])
        np.testing.assert_array_almost_equal(ds.date.data, expected[1])

    @pytest.mark.parametrize(
        ("variable", "expected"),
        [
            (
                "magnitude",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.9838, np.nan],
                            [-0.3835, -0.4497, -0.1234, -0.8852],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107907200, np.nan],
                            [1108339200, 1108512000, 1108857600, 1109116800],
                        ]
                    ),
                ),
            )
        ],
    )
    def test_last_negative(self, dataset, variable, expected):
        ds = sfilter.last_negative(dataset, variable=variable)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        np.testing.assert_array_almost_equal(ds.magnitude.data, expected[0])
        np.testing.assert_array_almost_equal(ds.date.data, expected[1])

    def test_filter_by_variable_error(self, dataset):
        with pytest.raises(ValueError, match="Invalid filter type."):
            sfilter.filter_by_variable(dataset, "negative")

    @pytest.mark.parametrize(
        ("filter_type", "expected"),
        [
            (
                "negative_of_first",
                (
                    np.array(
                        [
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, -0.4497, np.nan, -0.0617],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, 1108512000, np.nan, 1109030400],
                        ]
                    ),
                ),
            ),
            (
                "negative_of_last",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.9838, np.nan],
                            [-0.3835, np.nan, np.nan, np.nan],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107907200, np.nan],
                            [1108339200, np.nan, np.nan, np.nan],
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_filter_by_variable(self, dataset, filter_type, expected):
        dataset.magnitude.data = da.from_array(dataset.magnitude.data)
        dataset.date.data = da.from_array(dataset.date.data)
        ds = sfilter.filter_by_variable(dataset, filter_type)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        mag = ds.magnitude.compute()
        date = ds.date.compute()
        np.testing.assert_array_almost_equal(mag, expected[0])
        np.testing.assert_array_almost_equal(date, expected[1])

    @pytest.mark.parametrize(
        ("filter_type", "expected"),
        [
            (
                "first_negative",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.8798, np.nan],
                            [-0.3835, -0.4497, -0.1234, -0.0617],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107820800, np.nan],
                            [1108339200, 1108512000, 1108857600, 1109030400],
                        ]
                    ),
                ),
            ),
            (
                "last_negative",
                (
                    np.array(
                        [
                            [np.nan, -0.3283, -0.9838, np.nan],
                            [-0.3835, -0.4497, -0.1234, -0.8852],
                        ]
                    ),
                    np.array(
                        [
                            [np.nan, 1107561600, 1107907200, np.nan],
                            [1108339200, 1108512000, 1108857600, 1109116800],
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_filter_by_variable_chunks(self, dataset, filter_type, expected):
        dataset.magnitude.data = da.from_array(dataset.magnitude.data, chunks=(1, 2, 3))
        dataset.date.data = da.from_array(dataset.date.data, chunks=(1, 2, 3))
        ds = sfilter.filter_by_variable(dataset, filter_type)
        assert ds.magnitude.shape == (2, 4)
        assert ds.date.shape == (2, 4)
        mag = ds.magnitude.compute()
        date = ds.date.compute()
        np.testing.assert_array_almost_equal(mag, expected[0])
        np.testing.assert_array_almost_equal(date, expected[1])
