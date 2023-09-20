import numpy as np

__all__ = ["is_normalized", "normalize_array"]


def is_normalized(array: np.ndarray) -> bool:
    """Check if an array is normalized.

    Parameters
    ----------
    array : np.ndarray
        Array whose norm is checked.

    Returns
    -------
    bool
        True if the array norm is equal to 1, False otherwise.
    """
    data_norm = np.linalg.norm(array)
    return data_norm == 1.0


def normalize_array(array: np.ndarray) -> np.ndarray:
    """Scale array values so the euclidean norm is equal to 1.

    Parameters
    ----------
    array : np.ndarray
        Array whose values are scaled.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    data_norm = np.linalg.norm(array)
    return array / data_norm
