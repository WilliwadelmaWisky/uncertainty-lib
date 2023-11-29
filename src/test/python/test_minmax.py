import numpy as np
from src.main.python.minmax import calculate
from math import isclose


def test_calc_minmax_error() -> None:
    '''
    Tests for calc_minmax_error
    :return:
    '''

    val = np.array([1, 2])
    err = np.array([0.2, 0.5])

    result, config = calculate(lambda x, y: x + y, val, err)
    assert isclose(result, 0.7)

    result, config = calculate(lambda x, y: x - y, val, err)
    assert isclose(result, 0.7)

    val = np.array([1, 2, 3, 4])
    err = np.array([0.2, 0.5, 0.3, 0.1])

    result, config = calculate(lambda x, y, z, w: x + y + z + w, val, err)
    assert isclose(result, 1.1)
