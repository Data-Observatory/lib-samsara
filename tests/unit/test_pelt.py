import random

import numpy as np
import pytest

import samsara.pelt as pelt


class TestPelt:
    @pytest.mark.parametrize(("n", "bp"), [(15, 1), (15, 3), (15, 6)])
    def test_segment_metrics(self, n, bp):
        # array
        array = np.arange(n)
        # dates
        dates = np.array([np.datetime64("2010-01-01") + 10 * i for i in range(n)])
        # breaks
        break_idx = random.sample(range(1, n), bp)
        break_idx.sort()
        break_idx = np.array(break_idx)
        assert len(break_idx) == bp
        # run
        mean_mag, dates_frac = pelt.segment_metrics(array, dates, break_idx)
        assert len(mean_mag) == bp
        assert len(dates_frac) == bp
        assert (mean_mag > 0.0).all()

    @pytest.mark.parametrize(
        ("dates", "expected"),
        [
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

    def test_numba_segment_mean(self):
        array_t = np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    [1.0, 10.0, 10.0, 10.0, 10.5, 11.0, 11.0, 11.0],
                    [1.0, 10.0, 12.0, 8.0, 8.0, 8.5, 9.0, 9.0],
                ],
                [
                    [5.0, 10.0, 16.0, 20.0, 22.0, 20.0, 22.0, 21.0],
                    [7.0, 3.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0],
                    [6.0, 2.0, 4.0, 10.0, 12.0, 10.0, 12.0, 7.0],
                ],
            ],
            dtype=np.float32,
        )

        breaks_t = np.array(
            [
                [[np.nan, np.nan, np.nan], [1.0, np.nan, np.nan], [1.0, 3.0, np.nan]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 5.0], [1.0, 3.0, 7.0]],
            ],
            dtype=np.float32,
        )

        segment_mean_mag = np.full_like(breaks_t, np.nan)

        pelt.segment_mean(array_t, breaks_t, segment_mean_mag)

        expected = np.array(
            [
                [[np.nan, np.nan, np.nan], [9.5, np.nan, np.nan], [10.0, -2.5, np.nan]],
                [[5.0, 6.0, 5.0], [-4.0, 3.0, 7.0], [-3.0, 8.0, -4.0]],
            ],
            dtype=np.float32,
        )

        assert segment_mean_mag.shape == expected.shape
        np.testing.assert_array_equal(segment_mean_mag, expected)

    def test_numba_segment_dates(self):
        array_t = np.array(
            [
                [
                    [
                        2005.32,
                        2005.54,
                        2006.13,
                        2006.42,
                        2007.87,
                        2007.95,
                        2008.53,
                        2008.69,
                    ],
                    [
                        2005.62,
                        2005.75,
                        2006.09,
                        2006.17,
                        2007.35,
                        2007.62,
                        2008.57,
                        2008.81,
                    ],
                    [
                        2005.35,
                        2005.56,
                        2006.38,
                        2006.39,
                        2007.78,
                        2007.97,
                        2008.37,
                        2008.93,
                    ],
                ],
                [
                    [
                        2005.34,
                        2005.86,
                        2006.58,
                        2006.98,
                        2007.28,
                        2007.82,
                        2008.11,
                        2008.77,
                    ],
                    [
                        2005.58,
                        2005.75,
                        2006.22,
                        2006.77,
                        2007.36,
                        2007.56,
                        2008.13,
                        2008.84,
                    ],
                    [
                        2005.58,
                        2005.79,
                        2006.27,
                        2006.47,
                        2007.32,
                        2007.82,
                        2008.25,
                        2008.81,
                    ],
                ],
            ],
            dtype=np.float32,
        )

        breaks_t = np.array(
            [
                [[np.nan, np.nan, np.nan], [1.0, np.nan, np.nan], [1.0, 3.0, np.nan]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 5.0], [1.0, 3.0, 7.0]],
            ],
            dtype=np.float32,
        )

        segment_dates = np.full_like(breaks_t, np.nan)

        pelt.segment_dates(array_t, breaks_t, segment_dates)

        expected = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [2005.75, np.nan, np.nan],
                    [2005.56, 2006.39, np.nan],
                ],
                [
                    [2005.86, 2006.58, 2006.98],
                    [2005.75, 2006.22, 2007.56],
                    [2005.79, 2006.47, 2008.81],
                ],
            ],
            dtype=np.float32,
        )

        assert segment_dates.shape == expected.shape
        np.testing.assert_array_equal(segment_dates, expected)

    def test_block_segment_metrics(self):
        array = np.arange(48, dtype=np.float32).reshape((8, 2, 3))
        dates = np.array(
            [2005.58, 2005.79, 2006.27, 2006.47, 2007.32, 2007.82, 2008.25, 2008.81],
            dtype=np.float32,
        )
        break_idx = np.array(
            [
                [[np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[np.nan, np.nan, 3.0], [2.0, 2.0, 3.0]],
                [[np.nan, np.nan, np.nan], [3.0, 5.0, 7.0]],
            ]
        )

        assert array.shape == (8, 2, 3)
        assert dates.shape == (8,)
        assert break_idx.shape == (3, 2, 3)

        seg_mean, seg_date = pelt.block_segment_metrics(array, dates, break_idx)

        assert seg_mean.shape == (3, 2, 3)
        assert seg_date.shape == (3, 2, 3)

        expected_mean = np.array(
            [
                [[np.nan, 24.0, 9.0], [6.0, 6.0, 9.0]],
                [[np.nan, np.nan, 21.0], [6.0, 12.0, 18.0]],
                [[np.nan, np.nan, np.nan], [18.0, 18.0, 15.0]],
            ]
        )

        np.testing.assert_allclose(seg_mean, expected_mean, rtol=1e-02)

        expected_date = np.array(
            [
                [[np.nan, 2005.79, 2005.79], [2005.79, 2005.79, 2005.79]],
                [[np.nan, np.nan, 2006.47], [2006.27, 2006.27, 2006.47]],
                [[np.nan, np.nan, np.nan], [2006.47, 2007.82, 2008.81]],
            ]
        )

        np.testing.assert_allclose(seg_date, expected_date, rtol=1e-02)
