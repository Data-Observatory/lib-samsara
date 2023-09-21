import numpy as np
import pytest

from samsara.kernel import Kernel


class TestKernel:
    def test_empty_init(self):
        kernel = Kernel()
        assert isinstance(kernel.data, np.ndarray)
        assert kernel.name == "square"
        assert kernel.normalized is False
        assert kernel.shape == (5, 5)  # property

    def test_data_init(self):
        data = np.ones((7, 7))
        kernel = Kernel(data=data)
        assert isinstance(kernel.data, np.ndarray)
        assert kernel.name == "square"
        assert kernel.normalized is False
        assert kernel.shape == (7, 7)  # property
        assert kernel.data == pytest.approx(np.full_like(data, 1.0))

    def test_normalize_init(self):
        data = np.ones((7, 7))
        kernel = Kernel(data=data, normalized=True)
        assert isinstance(kernel.data, np.ndarray)
        assert kernel.name == "square"
        assert kernel.normalized is True
        assert kernel.shape == (7, 7)  # property
        assert kernel.data == pytest.approx(np.full_like(data, 0.142857143))

    def test_repr(self):
        kernel = Kernel()
        assert repr(kernel) == "5x5 square kernel."

    @pytest.mark.parametrize(
        ("kernel_1", "kernel_2", "normalize", "expected"),
        [
            (
                Kernel(np.ones((5, 5)), "square", False),
                Kernel(np.ones((5, 5)), "square", False),
                False,
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
            (
                Kernel(np.ones((5, 5)), "square", False),
                Kernel(np.ones((5, 5)), "square", False),
                True,
                Kernel(np.full((5, 5), 0.2), "custom", True),
            ),
            (
                Kernel(np.ones((5, 5)), "square", True),
                Kernel(np.ones((5, 5)), "square", True),
                False,
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            (
                Kernel(np.ones((5, 5)), "square", True),
                Kernel(np.ones((5, 5)), "square", True),
                True,
                Kernel(np.full((5, 5), 0.2), "custom", True),
            ),
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                Kernel(np.full((5, 5), 0.2), "square", True),
                False,
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
        ],
    )
    def test_add(self, kernel_1, kernel_2, normalize, expected):
        res = kernel_1.add(kernel_2, normalize)
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("rotation", "expected"),
        [
            (1, Kernel(np.array([[6, 3, 0], [7, 4, 1], [8, 5, 2]]), "custom", False)),
            (3, Kernel(np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]), "custom", False)),
            (2, Kernel(np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]]), "custom", False)),
            (4, Kernel(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), "custom", False)),
            (-1, Kernel(np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]), "custom", False)),
        ],
    )
    def test_rotate90(self, rotation, expected):
        kernel_data = np.arange(9).reshape((3, 3))
        kernel_1 = Kernel(kernel_data, "custom", False)
        kernel_2 = kernel_1.rotate90(rotation)
        assert kernel_2.data == pytest.approx(expected.data)
        assert kernel_2.normalized == pytest.approx(expected.normalized)
        assert kernel_2.name == pytest.approx(expected.name)

    # Test operators
    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 0.2),
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                2,
                Kernel(np.full((5, 5), 2.2), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                1.3,
                Kernel(np.full((5, 5), 1.5), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                Kernel(np.full((5, 5), 0.5), "square", False),
                Kernel(np.full((5, 5), 0.7), "custom", False),
            ),
        ],
    )
    def test_op_add(self, kernel_1, other, expected):
        res = kernel_1 + other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 0.1), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                2,
                Kernel(np.full((5, 5), -1.8), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                0.1,
                Kernel(np.full((5, 5), 0.1), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.5), "square", True),
                Kernel(np.full((5, 5), 1.2), "custom", False),
            ),
        ],
    )
    def test_op_sub(self, kernel_1, other, expected):
        res = kernel_1 - other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 2.3),
                Kernel(np.full((5, 5), 0.46), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                2,
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                1.1,
                Kernel(np.full((5, 5), 0.22), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.5), "square", True),
                Kernel(np.full((5, 5), 0.28), "custom", False),
            ),
        ],
    )
    def test_op_mul(self, kernel_1, other, expected):
        res = kernel_1 * other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                -2,
                Kernel(np.full((5, 5), -1.0), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                0.01,
                Kernel(np.full((5, 5), 20.0), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
        ],
    )
    def test_op_floordiv(self, kernel_1, other, expected):
        res = kernel_1 // other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.7),
                Kernel(np.full((5, 5), 0.28571428), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 1.3), "square", True),
                -2,
                Kernel(np.full((5, 5), -0.1), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 1.3), "square", False),
                3.2,
                Kernel(np.full((5, 5), 0.40625), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 2.3), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 3.833333), "custom", False),
            ),
        ],
    )
    def test_op_truediv(self, kernel_1, other, expected):
        res = kernel_1 / other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.15),
                Kernel(np.full((5, 5), 0.05), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                -2,
                Kernel(np.full((5, 5), -1.8), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                0.16,
                Kernel(np.full((5, 5), 0.04), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 0.2), "custom", False),
            ),
        ],
    )
    def test_op_mod(self, kernel_1, other, expected):
        res = kernel_1 % other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 0.8513399), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                2,
                Kernel(np.full((5, 5), 0.04), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                0.01,
                Kernel(np.full((5, 5), 0.9840344), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 1.2237052), "custom", False),
            ),
        ],
    )
    def test_op_pow(self, kernel_1, other, expected):
        res = kernel_1**other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 3), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 2, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_and(self, kernel_1, other, expected):
        res = kernel_1 & other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 5, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 3, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_xor(self, kernel_1, other, expected):
        res = kernel_1 ^ other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 7, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                8,
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 9, dtype=int), "square", False),
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_or(self, kernel_1, other, expected):
        res = kernel_1 | other
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 0.2),
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.5), "square", False),
                3,
                Kernel(np.full((5, 5), 3.5), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                Kernel(np.full((5, 5), 0.5), "square", False),
                Kernel(np.full((5, 5), 0.7), "custom", False),
            ),
        ],
    )
    def test_op_radd(self, kernel_1, other, expected):
        res = other + kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.5),
                Kernel(np.full((5, 5), 0.3), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.5), "square", False),
                3,
                Kernel(np.full((5, 5), 2.5), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                Kernel(np.full((5, 5), 0.5), "square", False),
                Kernel(np.full((5, 5), 0.3), "custom", False),
            ),
        ],
    )
    def test_op_rsub(self, kernel_1, other, expected):
        res = other - kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 2.3),
                Kernel(np.full((5, 5), 0.46), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                2,
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.5), "square", True),
                Kernel(np.full((5, 5), 0.28), "custom", False),
            ),
        ],
    )
    def test_op_rmul(self, kernel_1, other, expected):
        res = other * kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.4),
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                -2,
                Kernel(np.full((5, 5), -10.0), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 2.6), "square", False),
                Kernel(np.full((5, 5), 1.0), "custom", False),
            ),
        ],
    )
    def test_op_rfloordiv(self, kernel_1, other, expected):
        res = other // kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.5),
                Kernel(np.full((5, 5), 2.5), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.4), "square", False),
                -5,
                Kernel(np.full((5, 5), -12.5), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 2.6), "square", False),
                Kernel(np.full((5, 5), 1.8571428), "custom", False),
            ),
        ],
    )
    def test_op_rtruediv(self, kernel_1, other, expected):
        res = other / kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.15), "square", False),
                np.full((5, 5), 0.2),
                Kernel(np.full((5, 5), 0.05), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.3), "square", False),
                5,
                Kernel(np.full((5, 5), 0.2), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.2), "custom", False),
            ),
        ],
    )
    def test_op_rmod(self, kernel_1, other, expected):
        res = other % kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.1), "square", False),
                np.full((5, 5), 0.2),
                Kernel(np.full((5, 5), 0.8513399), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                2,
                Kernel(np.full((5, 5), 1.1486983), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 1.2237052), "custom", False),
            ),
        ],
    )
    def test_op_rpow(self, kernel_1, other, expected):
        res = other**kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 3), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 2, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_rand(self, kernel_1, other, expected):
        res = other & kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 5, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 3, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_rxor(self, kernel_1, other, expected):
        res = other ^ kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 7, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                8,
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 10, dtype=int), "square", False),
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_ror(self, kernel_1, other, expected):
        res = other | kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 0.2),
                Kernel(np.full((5, 5), 0.4), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                2,
                Kernel(np.full((5, 5), 2.2), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                Kernel(np.full((5, 5), 0.5), "square", False),
                Kernel(np.full((5, 5), 0.7), "custom", False),
            ),
        ],
    )
    def test_op_iadd(self, kernel_1, other, expected):
        kernel_1 += other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 0.1), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                0.1,
                Kernel(np.full((5, 5), 0.1), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.5), "square", True),
                Kernel(np.full((5, 5), 1.2), "custom", False),
            ),
        ],
    )
    def test_op_isub(self, kernel_1, other, expected):
        kernel_1 -= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                np.full((5, 5), 2.3),
                Kernel(np.full((5, 5), 0.46), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", True),
                1.1,
                Kernel(np.full((5, 5), 0.22), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.5), "square", True),
                Kernel(np.full((5, 5), 0.28), "custom", False),
            ),
        ],
    )
    def test_op_imul(self, kernel_1, other, expected):
        kernel_1 *= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                0.01,
                Kernel(np.full((5, 5), 20.0), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 2.0), "custom", False),
            ),
        ],
    )
    def test_op_ifloordiv(self, kernel_1, other, expected):
        kernel_1 //= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.7),
                Kernel(np.full((5, 5), 0.28571428), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 1.3), "square", False),
                3.2,
                Kernel(np.full((5, 5), 0.40625), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 2.3), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 3.833333), "custom", False),
            ),
        ],
    )
    def test_op_itruediv(self, kernel_1, other, expected):
        kernel_1 /= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.15),
                Kernel(np.full((5, 5), 0.05), "custom", False),
            ),
            # float
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                0.16,
                Kernel(np.full((5, 5), 0.04), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 0.2), "custom", False),
            ),
        ],
    )
    def test_op_imod(self, kernel_1, other, expected):
        kernel_1 %= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                np.full((5, 5), 0.1),
                Kernel(np.full((5, 5), 0.8513399), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 0.2), "square", False),
                2,
                Kernel(np.full((5, 5), 0.04), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 1.4), "square", False),
                Kernel(np.full((5, 5), 0.6), "square", False),
                Kernel(np.full((5, 5), 1.2237052), "custom", False),
            ),
        ],
    )
    def test_op_ipow(self, kernel_1, other, expected):
        kernel_1 **= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 3), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 2, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_iand(self, kernel_1, other, expected):
        kernel_1 &= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 4, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                2,
                Kernel(np.full((5, 5), 5, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 4, dtype=int), "square", False),
                Kernel(np.full((5, 5), 3, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_ixor(self, kernel_1, other, expected):
        kernel_1 ^= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            # array
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                np.full((5, 5), 3, dtype=int),
                Kernel(np.full((5, 5), 7, dtype=int), "custom", False),
            ),
            # int
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                8,
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
            # kernel
            (
                Kernel(np.full((5, 5), 7, dtype=int), "square", False),
                Kernel(np.full((5, 5), 9, dtype=int), "square", False),
                Kernel(np.full((5, 5), 15, dtype=int), "custom", False),
            ),
        ],
    )
    def test_op_ior(self, kernel_1, other, expected):
        kernel_1 |= other
        assert kernel_1.data == pytest.approx(expected.data)
        assert kernel_1.normalized == expected.normalized
        assert kernel_1.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "expected"),
        [
            (
                Kernel(np.full((3, 3), 1.0), normalized=False),
                Kernel(np.full((3, 3), -1.0), normalized=False),
            ),
            (
                Kernel(np.full((3, 3), 1.0), normalized=True),
                Kernel(np.full((3, 3), -0.2), normalized=True),
            ),
            (
                Kernel(np.array([[0, -1, 0], [1, -1, 1], [0, -1, 0]]), normalized=True),
                Kernel(
                    np.array(
                        [
                            [0, 0.4472136, 0],
                            [-0.4472136, 0.4472136, -0.4472136],
                            [0, 0.4472136, 0],
                        ]
                    ),
                    normalized=True,
                ),
            ),
        ],
    )
    def test_op_neg(self, kernel_1, expected):
        res = -kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "expected"),
        [
            (
                Kernel(np.full((3, 3), 1.0), normalized=False),
                Kernel(np.full((3, 3), 1.0), normalized=False),
            ),
            (
                Kernel(np.full((3, 3), 1.0), normalized=True),
                Kernel(np.full((3, 3), 0.2), normalized=True),
            ),
            (
                Kernel(np.array([[0, -1, 0], [1, -1, 1], [0, -1, 0]]), normalized=True),
                Kernel(
                    np.array(
                        [
                            [0, -0.4472136, 0],
                            [0.4472136, -0.4472136, 0.4472136],
                            [0, -0.4472136, 0],
                        ]
                    ),
                    normalized=True,
                ),
            ),
        ],
    )
    def test_op_pos(self, kernel_1, expected):
        res = +kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "expected"),
        [
            (
                Kernel(np.full((3, 3), 1.0), normalized=False),
                Kernel(np.full((3, 3), 1.0), normalized=False),
            ),
            (
                Kernel(np.full((3, 3), -1.0), normalized=True),
                Kernel(np.full((3, 3), 0.2), normalized=True),
            ),
            (
                Kernel(np.array([[0, -1, 0], [1, -1, 1], [0, -1, 0]]), normalized=True),
                Kernel(
                    np.array(
                        [
                            [0, 0.4472136, 0],
                            [0.4472136, 0.4472136, 0.4472136],
                            [0, 0.4472136, 0],
                        ]
                    ),
                    normalized=True,
                ),
            ),
        ],
    )
    def test_op_abs(self, kernel_1, expected):
        res = abs(kernel_1)
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "expected"),
        [
            (
                Kernel(np.full((3, 3), 1, dtype=int), normalized=False),
                Kernel(np.full((3, 3), -2, dtype=int), normalized=False),
            ),
            (
                Kernel(np.full((3, 3), -1, dtype=int), normalized=False),
                Kernel(np.full((3, 3), 0, dtype=int), normalized=False),
            ),
            (
                Kernel(
                    np.array([[0, -1, 0], [1, -1, 1], [0, -1, 0]], dtype=int),
                    normalized=False,
                ),
                Kernel(
                    np.array([[-1, 0, -1], [-2, 0, -2], [-1, 0, -1]], dtype=int),
                    normalized=False,
                ),
            ),
        ],
    )
    def test_op_invert(self, kernel_1, expected):
        res = ~kernel_1
        assert res.data == pytest.approx(expected.data)
        assert res.normalized == expected.normalized
        assert res.name == expected.name

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.7,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.7), normalized=False),
                np.ones((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_lt(self, kernel_1, other, expected):
        res = kernel_1 < other
        assert (res == expected).all()

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.7,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.ones((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_le(self, kernel_1, other, expected):
        res = kernel_1 <= other
        assert (res == expected).all()

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.0,
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.7), normalized=False),
                np.zeros((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_eq(self, kernel_1, other, expected):
        res = kernel_1 == other
        assert (res == expected).all()

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.0,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.7), normalized=False),
                np.ones((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_ne(self, kernel_1, other, expected):
        res = kernel_1 != other
        assert (res == expected).all()

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.0,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.7), normalized=False),
                np.zeros((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_ge(self, kernel_1, other, expected):
        res = kernel_1 >= other
        assert (res == expected).all()

    @pytest.mark.parametrize(
        ("kernel_1", "other", "expected"),
        [
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.0,
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                5.5,
                np.zeros((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                np.full((3, 3), 5.0),
                np.ones((3, 3), dtype=bool),
            ),
            (
                Kernel(np.full((3, 3), 5.5), normalized=False),
                Kernel(np.full((3, 3), 5.7), normalized=False),
                np.zeros((3, 3), dtype=bool),
            ),
        ],
    )
    def test_op_gt(self, kernel_1, other, expected):
        res = kernel_1 > other
        assert (res == expected).all()
