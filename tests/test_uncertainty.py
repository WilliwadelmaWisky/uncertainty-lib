import numpy as np
from src.uncertainty import minmax, standard
from math import isclose


def test_minmax() -> None:
    """
    A couple of tests for minmax-method
    :return: None
    """

    val = np.array([1, 2])
    err = np.array([0.2, 0.5])

    result = minmax(lambda x, y: x + y, val, err)
    assert isclose(result, 0.7)

    result = minmax(lambda x, y: x - y, val, err)
    assert isclose(result, 0.7)

    val = np.array([1, 2, 3, 4])
    err = np.array([0.2, 0.5, 0.3, 0.1])

    result = minmax(lambda x, y, z, w: x + y + z + w, val, err)
    assert isclose(result, 1.1)


def test_standard() -> None:
    """
    A couple of tests for standard-method
    :return: None
    """

    val = np.array([1, 2])
    err = np.array([0.2, 0.5])

    result = standard(lambda x, y: x + y, val, err)
    assert isclose(result, 0.5385, abs_tol=0.01)

    result = standard(lambda x, y: x**2 + 2 * y, val, err)
    assert isclose(result, 1.0770, abs_tol=0.01)

    result = standard(lambda x, y: 3 / 2 * x**4 + 2 / 5 * y**3, val, err)
    assert isclose(result, 2.6832, abs_tol=0.01)

    result = standard(lambda x, y: x**2 * y, val, err)
    assert isclose(result, 0.9433, abs_tol=0.01)
