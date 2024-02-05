from src.util import partial_df
from numpy import array
from math import isclose


def test_partial_df() -> None:
    """
    A couple of tests for partial_df
    :return:
    """
    f = lambda x, y: x**2 + y**2
    assert isclose(partial_df(f, array([1, 2]), 0), 2, abs_tol=0.01)
    assert isclose(partial_df(f, array([1, 2]), 1), 4, abs_tol=0.01)
