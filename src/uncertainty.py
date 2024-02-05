import numpy as np
from src.util import partial_df
from math import sqrt, floor


def minmax(func, val, err, full_output: bool = False):
    """
    Calculates an uncertainty using a min-max method, calculation is expensive because it uses bruteforce approach
    :param func: Function to apply min-max method
    :param val: All the variables required in a function arguments
    :param err: All the errors to variables, has to be of same size as value_data array
        (so even if a variable has an uncertainty of 0 it needs to be added to the array)
    :param full_output: Used to return extra information when set to TRUE
    :return: Calculated error. If full_output is set to TRUE returns a tuple of values.
        The first value is the calculated error, sthe second value is an array of (numbers) value
        configurations (0 = default, 1 = min, 2 = max). Index of the array indicates position of the argument
        (ex. f(x, y) -> [0, 1], meaning x=default, y=min)
    """
    val_count = len(val)
    config_count = 3 ** len(val)
    val_configs = np.zeros((config_count, val_count))
    fvalues = np.zeros(config_count)

    for i in range(0, config_count):
        values = np.zeros(val_count)
        for j in range(0, val_count):
            val_configs[i, j] = floor(i / 3 ** j) % 3
            match int(val_configs[i, j]):
                case 0:
                    values[j] = val[j]
                case 1:
                    values[j] = val[j] - err[j]
                case 2:
                    values[j] = val[j] + err[j]

        fvalues[i] = func(*values)

    refrence_fvalue = func(*val)
    ferrors = np.abs(fvalues - refrence_fvalue)
    max_error = np.max(ferrors)

    if not full_output:
        return max_error

    max_error_index = np.where(ferrors == max_error)[0][0]
    max_error_config = val_configs[max_error_index]
    return err, max_error_config


def standard(func, val, err) -> float:
    """
    Calculate uncertainty with a standard uncertainty propagation method
    :param func: Function
    :param val: Values of function parameters (list or tuple).
        Make sure to
    :param err: Errors of values (list or tuple).
        If a value has 0 uncertainty, it must still be included in the list
    :return: Calculated error
    """
    error = 0.0
    for i in range(0, len(val)):
        error += (partial_df(func, val, i) * err[i]) ** 2

    return sqrt(error)

