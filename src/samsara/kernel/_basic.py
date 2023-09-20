import numpy as np

from ._kernel import Kernel
from ._utils import normalize_array

__all__ = [
    "circle",
    "cross",
    "custom",
    "ex",
    "octagon",
    "rectangle",
    "rhombus",
    "square",
]


def circle(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    data = np.where(distance_from_center <= radius, 1.0, 0.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="circle", normalized=True)

    return Kernel(data=data, name="circle", normalized=False)


def cross(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    data = np.zeros((side, side))
    data[:, radius] = 1.0
    data[radius, :] = 1.0

    if normalize:
        return Kernel(data=normalize_array(data), name="cross", normalized=True)

    return Kernel(data=data, name="cross", normalized=False)


def custom(data: np.ndarray, normalize: bool = False) -> Kernel:
    if normalize:
        return Kernel(data=normalize_array(data), name="custom", normalized=True)

    return Kernel(data=data, name="custom", normalized=False)


def ex(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    data = np.zeros((side, side))
    np.fill_diagonal(data, 1.0)
    np.fill_diagonal(np.flipud(data), 1.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="ex", normalized=True)

    return Kernel(data=data, name="ex", normalized=False)


def octagon(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    x, y = np.meshgrid(
        np.abs(np.arange(side) - radius), np.abs(np.arange(side) - radius)
    )
    limit = radius + radius // 2
    data = np.where(x + y <= limit, 1.0, 0.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="octagon", normalized=True)

    return Kernel(data=data, name="octagon", normalized=False)


def rectangle(x_radius: int, y_radius: int, normalize: bool = False) -> Kernel:
    x_side = 2 * x_radius + 1
    y_side = 2 * y_radius + 1
    data = np.ones((x_side, y_side))

    if normalize:
        return Kernel(data=normalize_array(data), name="rectangle", normalized=True)

    return Kernel(data=data, name="rectangle", normalized=False)


def rhombus(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    x, y = np.meshgrid(
        np.abs(np.arange(side) - radius), np.abs(np.arange(side) - radius)
    )
    data = np.where(x + y <= radius, 1.0, 0.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="rhombus", normalized=True)

    return Kernel(data=data, name="rhombus", normalized=False)


def square(radius: int, normalize: bool = False) -> Kernel:
    side = 2 * radius + 1
    data = np.ones((side, side))

    if normalize:
        return Kernel(data=normalize_array(data), name="square", normalized=True)

    return Kernel(data=data, name="square", normalized=False)
