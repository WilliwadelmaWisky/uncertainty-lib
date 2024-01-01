from src.util import calc_partial_derivatives_at, all_combinations


def test_calc_partial_derivatives_at() -> None:
    '''
    Tests for calc_partial_derivatives_at
    :return:
    '''

    point = (3, 2)
    derivatives = calc_partial_derivatives_at(lambda x, y: x**2 + 2*y, 'x y', point)
    assert derivatives[0] == 6
    assert derivatives[1] == 2


def test_all_combinations() -> None:
    '''
    Tests for all_combinations
    :return:
    '''

    assert len(all_combinations(1)) == 3
    assert len(all_combinations(2)) == 9
    assert len(all_combinations(3)) == 27
