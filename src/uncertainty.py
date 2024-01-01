import numpy as np
import sympy as sy
from src.util import calc_partial_derivatives_at, calc_all_values, all_combinations


def calc_minmax(func, value_data, error_data):
    '''
    Calculates an uncertainty using a min-max method, calculation is expensive because it uses bruteforce approach
    :param func: Function to apply min-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    (so even if a variable has an uncertainty of 0 it needs to be added to the array)
    :return: Tuple of (float, Array), first value is the calculated error and the second value
    is an array of value configurations (0 = default, 1 = min, 2 = max). Index of the array indicates
    position of the argument
    '''

    ref = func(*value_data)
    all_values = calc_all_values(func, value_data, error_data)
    all_errors = np.abs(all_values - ref)
    err = np.max(all_errors)

    max_index = np.where(all_errors == err)[0][0]
    all_configs = all_combinations(len(value_data))
    config = all_configs[max_index]

    return err, config


def calc_standard(func, symbols: str, value_data, error_data):
    '''
    Calculates an uncertainty using a standard uncertainty propagation method
    :param func: Function to calulate (requires sympy fuctions ex. sympy.exp() rather than numpy.exp())
    :param symbols: String of all symbols in the function separated with whitespace, ex. 'x y z'
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    (so even if a variable has an uncertainty of 0 it needs to be added to the array)
    :return: Calculated uncertainty
    '''

    derivative_values = calc_partial_derivatives_at(func, symbols, value_data)
    variance = 0

    for i in range(len(derivative_values)):
        variance += derivative_values[i]**2 * error_data[i]**2

    error = sy.sqrt(variance)
    return error
