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
