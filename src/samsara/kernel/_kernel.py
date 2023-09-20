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
    data: np.ndarray = field(default_factory=_default_data)
    name: str = field(default="square")
    normalized: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.normalized:
            data_norm = np.linalg.norm(self.data)
            if data_norm != 1.0:
                self.data = self.data / data_norm

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def add(self, kernel: Kernel, normalize: bool = False) -> Kernel:
        add_kernels = self + kernel

        # Normalize data if normalized flag is set
        if normalize:
            add_kernels.normalized = True
            if not is_normalized(add_kernels.data):
                add_kernels.data = normalize_array(add_kernels.data)

        return add_kernels

    def rotate(self, rotation: int) -> Kernel:
        raise NotImplementedError

    def __repr__(self):
        shape = str(self.data.shape).replace(", ", "x")[1:-1]
        return f"{shape} {self.name} kernel."

    def __binary_operation(self, other: Any, operation: callable) -> np.ndarray:
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
