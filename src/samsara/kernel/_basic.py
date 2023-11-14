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
    """Circle shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as a circle.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.circle(3)
    >>> x.data
    array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 1., 0., 0., 0.]])
    >>> y = skernel.circle(2, True)
    >>> y.data
    array([[0.       , 0.       , 0.2773501, 0.       , 0.       ],
           [0.       , 0.2773501, 0.2773501, 0.2773501, 0.       ],
           [0.2773501, 0.2773501, 0.2773501, 0.2773501, 0.2773501],
           [0.       , 0.2773501, 0.2773501, 0.2773501, 0.       ],
           [0.       , 0.       , 0.2773501, 0.       , 0.       ]])
    """
    side = 2 * radius + 1
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    data = np.where(distance_from_center <= radius, 1.0, 0.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="circle", normalized=True)

    return Kernel(data=data, name="circle", normalized=False)


def cross(radius: int, normalize: bool = False) -> Kernel:
    """Cross shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as a cross.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.cross(3)
    >>> x.data
    array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.]])
    """
    side = 2 * radius + 1
    data = np.zeros((side, side))
    data[:, radius] = 1.0
    data[radius, :] = 1.0

    if normalize:
        return Kernel(data=normalize_array(data), name="cross", normalized=True)

    return Kernel(data=data, name="cross", normalized=False)


def custom(data: np.ndarray, normalize: bool = False) -> Kernel:
    """Custom kernel.

    Parameters
    ----------
    data : np.ndarray
        The data array of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with a custom data array.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> import numpy as np
    >>> data = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    >>> x = skernel.custom(data)
    >>> x.data
    array([[1, 0, 1],
           [0, 1, 1],
           [0, 0, 0]])
    """
    if normalize:
        return Kernel(data=normalize_array(data), name="custom", normalized=True)

    return Kernel(data=data, name="custom", normalized=False)


def ex(radius: int, normalize: bool = False) -> Kernel:
    """Ex (x) shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as an ex (x).

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.ex(3)
    >>> x.data
    array([[1., 0., 0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 0., 0., 1.]])
    """
    side = 2 * radius + 1
    data = np.zeros((side, side))
    np.fill_diagonal(data, 1.0)
    np.fill_diagonal(np.flipud(data), 1.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="ex", normalized=True)

    return Kernel(data=data, name="ex", normalized=False)


def octagon(radius: int, normalize: bool = False) -> Kernel:
    """Octagon shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as an octagon.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.octagon(3)
    >>> x.data
    array([[0., 0., 1., 1., 1., 0., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 1., 1., 1., 0., 0.]])
    """
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
    """Rectangular shaped kernel.

    Parameters
    ----------
    x_radius : int
        The horizontal radius of the kernel to generate.
    y_radius : int
        The vertical radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as a rectangle.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.rectangle(2, 3)
    >>> x.data
    array([[1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.]])
    """
    x_side = 2 * x_radius + 1
    y_side = 2 * y_radius + 1
    data = np.ones((x_side, y_side))

    if normalize:
        return Kernel(data=normalize_array(data), name="rectangle", normalized=True)

    return Kernel(data=data, name="rectangle", normalized=False)


def rhombus(radius: int, normalize: bool = False) -> Kernel:
    """Rhombus (diamond) shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as a rhombus (diamond).

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.rhombus(3)
    >>> x.data
    array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 1., 1., 0., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 1., 1., 1., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.]])
    """
    side = 2 * radius + 1
    x, y = np.meshgrid(
        np.abs(np.arange(side) - radius), np.abs(np.arange(side) - radius)
    )
    data = np.where(x + y <= radius, 1.0, 0.0)

    if normalize:
        return Kernel(data=normalize_array(data), name="rhombus", normalized=True)

    return Kernel(data=data, name="rhombus", normalized=False)


def square(radius: int, normalize: bool = False) -> Kernel:
    """Square shaped kernel.

    Parameters
    ----------
    radius : int
        The radius of the kernel to generate.
    normalize : bool, optional
        Normalize the kernel data values so its euclidean norm is equal to 1, by default False.

    Returns
    -------
    Kernel
        Kernel with data array shaped as a square.

    Examples
    --------
    >>> import samsara.kernel as skernel
    >>> x = skernel.square(3)
    >>> x.data
    array([[1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.]])
    """
    side = 2 * radius + 1
    data = np.ones((side, side))

    if normalize:
        return Kernel(data=normalize_array(data), name="square", normalized=True)

    return Kernel(data=data, name="square", normalized=False)
