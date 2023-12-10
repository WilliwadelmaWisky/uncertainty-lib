import numpy as np
import sympy as sy


def calc_partial_derivatives(func, symbols: str):
    '''
    Calculates an uncertainty using a standard uncertainty propagation method
    :param func: Function to calulate (requires sympy fuctions ex. sympy.exp() rather than numpy.exp())
    :param symbols:
    :return: Calculated uncertainty
    '''

    symbols = sy.symbols(symbols)
    f = func(*symbols)
    derivatives = []

    for i in range(len(symbols)):
        derivatives.append(f.diff(symbols[i]))

    return derivatives


def calc_partial_derivatives_at(func, symbols: str, point):
    '''
    Calculates an uncertainty using a standard uncertainty propagation method
    :param func: Function to calulate (requires sympy fuctions ex. sympy.exp() rather than numpy.exp())
    :param symbols:
    :param point: A tuple or an array of input values
    :return: Calculated uncertainty
    '''

    point_data = []
    s = sy.symbols(symbols)
    for i in range(len(s)):
        point_data.append((s[i], point[i]))

    derivatives = calc_partial_derivatives(func, symbols)
    results = []
    for i in range(len(derivatives)):
        results.append(derivatives[i].subs(point_data))

    return results


def calc_partial_derivative_at(func, symbols: str, point, axis: int):
    '''
    Calculates an uncertainty using a standard uncertainty propagation method
    :param func: Function to calulate (requires sympy fuctions ex. sympy.exp() rather than numpy.exp())
    :param symbols:
    :param point: A tuple or an array of input values
    :param axis: An integer value of the index of the derivative (axis = 0, 1, 2...). 0 means the first argument,
    1 second argument and so on.
    :return: Calculated uncertainty
    '''

    values = calc_partial_derivatives_at(func, symbols, point)
    return values[axis]
