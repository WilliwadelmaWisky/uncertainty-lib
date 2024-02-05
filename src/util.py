from numpy import copy


def partial_df(f, x, axis: int, h: float = 1e-8) -> float:
    """
    Calculate (numerical) partial derivate at a certain point
    :param f: Function, same amount of arguments as the x has components
    :param x: Point (list or tuple)
    :param axis: Index of the partial derivative (ex. f(x, y), axis=0 -> df/dx, axis=1 -> df/dy)
    :param h: Precision of the calculation, small value
    :return: Value of the partial derivative
    """
    xplus = copy(x).astype(float)
    xminus = copy(x).astype(float)
    xplus[axis] += h
    xminus[axis] -= h
    return 0.5 * (f(*xplus) - f(*xminus)) / h
