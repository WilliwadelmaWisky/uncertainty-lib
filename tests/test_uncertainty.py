import numpy as np
from src.uncertainty import calc_minmax, calc_standard, standard
from math import isclose


def test_calc_minmax() -> None:
    '''
    Tests for calc_minmax
    :return:
    '''

    val = np.array([1, 2])
    err = np.array([0.2, 0.5])

    result, config = calc_minmax(lambda x, y: x + y, val, err)
    assert isclose(result, 0.7)

    result, config = calc_minmax(lambda x, y: x - y, val, err)
    assert isclose(result, 0.7)

    val = np.array([1, 2, 3, 4])
    err = np.array([0.2, 0.5, 0.3, 0.1])

    result, config = calc_minmax(lambda x, y, z, w: x + y + z + w, val, err)
    assert isclose(result, 1.1)


def test_calc_standard() -> None:
    '''
    Tests for calc_standard
    :return:
    '''

    val = np.array([1, 2], dtype=float)
    err = np.array([0.2, 0.5])

    result = calc_standard(lambda x, y: x + y, 'x y', val, err)
    assert isclose(result, 0.5385, abs_tol=0.01)

    result = standard(lambda x: x[0] + x[1], val, err)
    assert isclose(result, 0.5385, abs_tol=0.01)

    result = calc_standard(lambda x, y: x**2 + 2*y, 'x y', val, err)
    assert isclose(result, 1.0770, abs_tol=0.01)

    result = calc_standard(lambda x, y: 3/2*x**4 + 2/5*y**3, 'x y', val, err)
    assert isclose(result, 2.6832, abs_tol=0.01)

    result = calc_standard(lambda x, y: x**2 * y, 'x y', val, err)
    assert isclose(result, 0.9433, abs_tol=0.01)
