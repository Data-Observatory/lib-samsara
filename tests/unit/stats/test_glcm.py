import math

import numpy as np
import pytest

import samsara.stats.glcm as glcm


class TestGLCM:
    @pytest.mark.parametrize(
        ("normed", "nan_supression", "expected"),
        [
            (
                True,
                0,
                np.stack(
                    (
                        np.vstack((np.ones((1, 2, 2)), np.zeros((3, 2, 2)))),
                        *[np.zeros((4, 2, 2))] * 3,
                    )
                ),
            ),
            (
                False,
                0,
                np.stack(
                    (
                        np.vstack((np.ones((1, 2, 2)), np.zeros((3, 2, 2)))),
                        *[np.zeros((4, 2, 2))] * 3,
                    )
                ),
            ),
            (True, 1, np.zeros((4, 4, 2, 2))),
            (False, 1, np.zeros((4, 4, 2, 2))),
            (True, 2, np.zeros((4, 4, 2, 2))),
            (False, 2, np.zeros((4, 4, 2, 2))),
            (True, 3, np.zeros((3, 3, 2, 2))),
            (False, 3, np.zeros((3, 3, 2, 2))),
            (
                True,
                4,
                np.stack(
                    (
                        np.vstack((np.ones((1, 2, 2)), np.zeros((3, 2, 2)))),
                        *[np.zeros((4, 2, 2))] * 3,
                    )
                ),
            ),
            (
                False,
                4,
                np.stack(
                    (
                        np.vstack((np.ones((1, 2, 2)), np.zeros((3, 2, 2)))),
                        *[np.zeros((4, 2, 2))] * 3,
                    )
                ),
            ),
        ],
    )
    def test_matrix_rescale_normed_zero_division(
        self, normed, nan_supression, expected
    ):
        rescale_normed = True
        array = np.zeros((4, 4), dtype=np.uint8)
        distances = [1, 2]
        angles = [0, math.pi]
        levels = 4

        res = glcm.matrix(
            array,
            distances,
            angles,
            levels,
            symmetric=False,
            normed=normed,
            nan_supression=nan_supression,
            rescale_normed=rescale_normed,
        )

        assert res.ndim == 4
        assert res.shape == expected.shape
        np.testing.assert_array_almost_equal(res, expected)
