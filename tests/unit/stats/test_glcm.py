import math

import dask.array as da
import numpy as np
import pytest
import xarray as xr

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

    @pytest.mark.parametrize(
        ("normed", "nan_supression", "expected"),
        [
            (
                True,
                0,
                np.stack(
                    (
                        np.vstack((np.full((1, 2, 2), 0.25), np.zeros((3, 2, 2)))),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((2, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((3, 2, 2)), np.full((1, 2, 2), 0.25))),
                    )
                ),
            ),
            (
                False,
                0,
                np.stack(
                    (
                        np.vstack((np.full((1, 2, 2), 0.25), np.zeros((3, 2, 2)))),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((2, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((3, 2, 2)), np.full((1, 2, 2), 0.25))),
                    )
                ),
            ),
            (
                False,
                1,
                np.stack(
                    (
                        np.zeros((4, 2, 2)),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((2, 2, 2)),
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((3, 2, 2)), np.full((1, 2, 2), 1 / 3))),
                    )
                ),
            ),
            (
                False,
                2,
                np.stack(
                    (
                        np.zeros((4, 2, 2)),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((2, 2, 2)),
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((3, 2, 2)), np.full((1, 2, 2), 1 / 3))),
                    )
                ),
            ),
            (
                False,
                3,
                np.stack(
                    (
                        np.vstack(
                            (
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 1 / 3),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((2, 2, 2)), np.full((1, 2, 2), 1 / 3))),
                    )
                ),
            ),
            (
                False,
                4,
                np.stack(
                    (
                        np.vstack((np.full((1, 2, 2), 0.25), np.zeros((3, 2, 2)))),
                        np.vstack(
                            (
                                np.zeros((1, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((2, 2, 2)),
                            )
                        ),
                        np.vstack(
                            (
                                np.zeros((2, 2, 2)),
                                np.full((1, 2, 2), 0.25),
                                np.zeros((1, 2, 2)),
                            )
                        ),
                        np.vstack((np.zeros((3, 2, 2)), np.full((1, 2, 2), 0.25))),
                    )
                ),
            ),
        ],
    )
    def test_matrix_rescale_normed(self, normed, nan_supression, expected):
        rescale_normed = True
        array = np.array(
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.uint8
        )
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

    @pytest.mark.parametrize(("advanced_idx"), [True, False])
    def test_properties_plus_minus(self, advanced_idx):
        array = np.arange(4, dtype=np.float64).reshape((2, 2))
        p = array / array.sum()
        xpy = np.full(4, fill_value=0, dtype=np.double)
        xmy = np.full(2, fill_value=0, dtype=np.double)

        xpy, xmy = glcm.plus_minus(p, xpy, xmy, advanced_idx=advanced_idx)
        np.testing.assert_array_almost_equal(xpy, np.array([0, 0.5, 0.5, 0]))
        np.testing.assert_array_almost_equal(xmy, np.array([0.5, 0.5]))

    def test_properties_plus_minus_error(self):
        array = np.arange(6, dtype=np.float64).reshape((2, 3))
        p = array / array.sum()
        xpy = np.full(4, fill_value=0, dtype=np.double)
        xmy = np.full(2, fill_value=0, dtype=np.double)
        with pytest.raises(ValueError, match="p array is not square."):
            glcm.plus_minus(p, xpy, xmy)

    @pytest.mark.parametrize(
        ("data", "okwarg"),
        [
            (np.arange(4).repeat(4, axis=0).reshape((4, 4)), {}),
            (da.from_array(np.arange(4).repeat(4, axis=0).reshape((4, 4))), {}),
            (np.arange(4).repeat(4, axis=0).reshape((4, 4)), {"advanced_idx": True}),
            (
                da.from_array(np.arange(4).repeat(4, axis=0).reshape((4, 4))),
                {"advanced_idx": True},
            ),
            (np.arange(4).repeat(4, axis=0).reshape((4, 4)), {"nan_supression": 1}),
            (
                da.from_array(np.arange(4).repeat(4, axis=0).reshape((4, 4))),
                {"nan_supression": 1},
            ),
            (
                np.arange(4).repeat(4, axis=0).reshape((4, 4)),
                {"skip_nan": False},
            ),
            (
                da.from_array(np.arange(4).repeat(4, axis=0).reshape((4, 4))),
                {"skip_nan": False},
            ),
            (
                np.arange(4).repeat(4, axis=0).reshape((4, 4)),
                {
                    "nan_supression": 1,
                    "skip_nan": False,
                    "rescale_normed": True,
                    "advanced_idx": True,
                },
            ),
            (
                da.from_array(np.arange(4).repeat(4, axis=0).reshape((4, 4))),
                {
                    "nan_supression": 1,
                    "skip_nan": False,
                    "rescale_normed": True,
                    "advanced_idx": True,
                },
            ),
        ],
    )
    def test_glcm_textures_run(self, data, okwarg):
        array = xr.DataArray(
            data=data, dims=("x", "y"), coords={"x": np.arange(4), "y": np.arange(4)}
        )
        distances = [1, 2]
        angles = [0, math.pi]
        levels = 4
        radius = 2
        bgkwargs = {
            "distances": distances,
            "angles": angles,
            "levels": levels,
            **okwarg,
        }
        res = glcm.glcm_textures(array=array, radius=radius, n_feats=7, **bgkwargs)
        assert res.shape == (7, 4, 4)
        assert set(res.dims) - {"prop", "y", "x"} == set()
        assert set(res.coords) - {"prop", "y", "x"} == set()

    @pytest.mark.parametrize(("data"), [np.zeros((2, 2, 2)), da.ones((3, 4, 5, 6))])
    def test_glcm_textures_error_ndim(self, data):
        array = xr.DataArray(data=data)
        with pytest.raises(ValueError, match="Expected 2-dimensional data, "):
            glcm.glcm_textures(array, radius=2)

    @pytest.mark.parametrize(("data"), [np.ones((5, 5))])
    def test_glcm_textures_error_type(self, data):
        with pytest.raises(TypeError, match="Invalid type of data array in input."):
            glcm.glcm_textures(data, radius=2)
