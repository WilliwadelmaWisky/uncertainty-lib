import math
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


def all_combinations(arg_count: int):
    '''
    Get all the combinations of how amount of arguments can be arraged, all the arguments can have 3 different values
    (min, default, max).
    :param arg_count: Amount of arguments
    :return: Numpy matrix, combinations are stored in rows, value of a single cell means which type of value it is
    (0 = default, 1 = min, 2 = max).
    '''

    total_size = 3 ** arg_count
    config = np.zeros((total_size, arg_count))

    for x in range(0, total_size):
        for y in range(0, arg_count):
            config[x, y] = math.floor(x / 3 ** y) % 3

    return config


def get_masked_value(val: float, err: float, mask: int) -> float:
    '''
    Change value based of mask.
    :param val: Value
    :param err: Error of value
    :param mask: Mask (1 = min, 2 = max), otherwise default
    :return: Masked value
    '''

    if mask == 1:
        return val - err
    elif mask == 2:
        return val + err
    return val


def iterate_all(value_data, error_data, on_iterate) -> None:
    '''
    Iterates over all the possible configurations used in mix-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :param on_iterate: Function to be called on every iteration, Arguments is a tuple of (Array, int, Array),
    first argument is an integer refering to the iteration number (0 ... N) and the second argument is an array of
    all the values calculated in a single iteration, the third argument is an array of a value configurations
    (0 = default, 1 = min, 2 = max).
    :return: None
    '''

    data_size = len(value_data)
    config = all_combinations(data_size)
    values = np.zeros(data_size)

    for x in range(0, config.shape[0]):
        for y in range(0, config.shape[1]):
            value_mask = int(config[x, y])
            values[y] = get_masked_value(value_data[y], error_data[y], value_mask)

        on_iterate((x, values, config[x]))


def calc_all_values(func, value_data, error_data):
    '''
    Calculates all the possible configurations for min-max method
    :param func: Function to calculate configurations from
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :return: Numpy array (1d) containing all the calculated values
    '''

    total_size = 3 ** len(value_data)
    results = np.zeros(total_size)

    def assign_calc(index, values):
        results[index] = func(*values)

    iterate_all(value_data, error_data, lambda values: assign_calc(values[0], values[1]))
    return results


def partial_df(f, x, axis: int, h: float = 1e-8) -> float:
    """
    Calculate (numerical) partial derivate at a certain point
    :param f: Function
    :param x: Point (list or tuple)
    :param axis: Index of the partial derivative (ex. f(x, y), axis=0 -> df/dx, axis=1 -> df/dy)
    :param h: Precision of the calculation, small value
    :return: Value of the partial derivative
    """
    xplus, xminus = np.copy(x), np.copy(x)
    xplus[axis] += h
    xminus[axis] -= h
    return 0.5 * (f(xplus) - f(xminus)) / h
