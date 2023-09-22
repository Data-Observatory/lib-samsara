from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import is_normalized, normalize_array

__all__ = ["Kernel"]


def _default_data() -> np.ndarray:
    return np.ones((5, 5))


@dataclass(init=True, repr=False)
class Kernel:
    """Kernel for neighborhood based image computation.

    A Kernel is an object to specify the neighborhood of a pixel. The kernel defines the size and
    shape of the neighborhood, and also defines the weight of each element in the neighborhood.

    Parameters
    ----------
    data: np.ndarray, optional
        Kernel data array.
    name: str, optional
        Name of the type of kernel.
    normalized: bool, optional
        Flag to indicate if the euclidean norm of the data array is equal to 1, by default False.
    """

    data: np.ndarray = field(default_factory=_default_data)
    """Kernel data array (np.ndarray).
    """
    name: str = field(default="square")
    """Name of the type of kernel (str).
    """
    normalized: bool = field(default=False)
    """Flag to indicate if the euclidean norm of the data array is equal to 1 (bool).
    """
    # Override numpy array behavior to avoid entering ufunc
    __array_ufunc__ = None

    def __post_init__(self) -> None:
        if self.normalized:
            data_norm = np.linalg.norm(self.data)
            if data_norm != 1.0:
                self.data = self.data / data_norm

    @property
    def shape(self) -> tuple:
        """Shape of the kernel data array (tuple, read-only)."""
        return self.data.shape

    def add(self, kernel: Kernel, normalize: bool = False) -> Kernel:
        """Add two kernels elementwise.

        Elementwise sum of data from two kernels. The shape of the data of both kernels must be
        broadcastable.
        Currently the sum of kernels with different shapes is not supported.

        Parameters
        ----------
        kernel : Kernel
            The kernel to be added.
        normalize : bool, optional
            Normalize the sum of both kernels, by default False

        Returns
        -------
        Kernel
            The sum of both kernels elementwise.

        Examples
        --------
        >>> from samsara.kernel import Kernel
        >>> import numpy as np
        >>> x = Kernel(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), "custom", False)
        >>> y = Kernel(np.ones((3, 3)), "square", False)
        >>> x.add(y).data
        array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]])
        """
        add_kernels = self + kernel

        # Normalize data if normalized flag is set
        if normalize:
            add_kernels.normalized = True
            if not is_normalized(add_kernels.data):
                add_kernels.data = normalize_array(add_kernels.data)

        return add_kernels

    def rotate90(self, rotation: int) -> Kernel:
        """Rotates the data array of a kernel.

        Creates a new kernel with rotated data. The rotations are 90 degree rotations.

        Parameters
        ----------
        rotation : int
            Number of 90 degrees clockwise rotation to make.

        Returns
        -------
        Kernel
            A new kernel with rotated data.

        Examples
        --------
        >>> from samsara.kernel import Kernel
        >>> import numpy as np
        >>> x = Kernel(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), "custom", False)
        >>> x.rotate90(1).data
        array([[6, 3, 0],
               [7, 4, 1],
               [8, 5, 2]])
        """
        # copied to generate new data
        new_data = np.rot90(self.data, -rotation).copy()
        return self.__class__(new_data, self.name, self.normalized)

    def __repr__(self):
        shape = str(self.data.shape).replace(", ", "x")[1:-1]
        return f"{shape} {self.name} kernel."

    def __binary_operation(self, other: Any, operation: callable) -> np.ndarray:
        # Operation of different data types with Kernel
        if isinstance(other, Kernel):
            new_data = operation(self.data, other.data)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            new_data = operation(self.data, other)
        else:
            raise TypeError
        return new_data

    # Binary operators
    def __add__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x + y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __sub__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x - y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __mul__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x * y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __floordiv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x // y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __truediv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x / y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __mod__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x % y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __pow__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x**y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __and__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x & y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __xor__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x ^ y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __or__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x | y)
        return self.__class__(data=new_data, name="custom", normalized=False)

    # Right side operators
    def __radd__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y + x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rsub__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y - x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rmul__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y * x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rfloordiv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y // x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rtruediv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y / x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rmod__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y % x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rpow__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y**x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rand__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y & x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __rxor__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y ^ x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    def __ror__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: y | x)
        return self.__class__(data=new_data, name="custom", normalized=False)

    # In place operators
    def __iadd__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x + y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __isub__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x - y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __imul__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x * y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __idiv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x / y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __ifloordiv__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x // y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __imod__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x % y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __ipow__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x**y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __iand__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x & y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __ixor__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x ^ y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    def __ior__(self, other: Any) -> Kernel:
        new_data = self.__binary_operation(other, lambda x, y: x | y)
        self.data = new_data
        self.name = "custom"
        self.normalized = False
        return self

    # Unary operators
    def __neg__(self) -> Kernel:
        new_data = -self.data
        return self.__class__(data=new_data, name=self.name, normalized=self.normalized)

    def __pos__(self) -> Kernel:
        new_data = +self.data
        return self.__class__(data=new_data, name=self.name, normalized=self.normalized)

    def __abs__(self) -> Kernel:
        new_data = np.abs(self.data)
        return self.__class__(data=new_data, name=self.name, normalized=self.normalized)

    def __invert__(self) -> Kernel:
        new_data = np.invert(self.data)
        normalized = is_normalized(new_data)
        return self.__class__(data=new_data, name=self.name, normalized=normalized)

    # Comparison operators
    def __lt__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x < y)
        return compared

    def __le__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x <= y)
        return compared

    def __eq__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x == y)
        return compared

    def __ne__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x != y)
        return compared

    def __ge__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x >= y)
        return compared

    def __gt__(self, other: Any) -> np.ndarray:
        compared = self.__binary_operation(other, lambda x, y: x > y)
        return compared
