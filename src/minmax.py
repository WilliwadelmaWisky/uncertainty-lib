import math
from typing import Final
import numpy as np


DEFAULT: Final[int] = 0
MIN: Final[int] = 1
MAX: Final[int] = 2


def calc_minmax_error(func, value_data, error_data) -> tuple:
    '''
    Calculates an uncertainty using a min-max method, calculation is expensive
    :param func: Function to apply min-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :return: Tuple of (float, Array), first value is the calculated error and the second value
    is an array of value configurations (0 = default, 1 = min, 2 = max). Index of the array indicates
    position of the argument
    '''

    result = 0
    data_size = len(value_data)
    value_config = np.zeros(data_size)
    ref = func(*value_data)

    def calc_result(values, config):
        nonlocal result
        temp = math.fabs(func(*values) - ref)
        if temp > result:
            result = temp
            for i in range(0, data_size):
                value_config[i] = config[i]

    __iterate_all_values(value_data, error_data, lambda values: calc_result(values[0], values[2]))
    # print(result)
    return result, value_config


def calc_minmax_all(func, value_data, error_data):
    '''
    Calculates all the possible configurations using min-max method
    :param func: Function to apply min-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :return: Numpy array (1d) containing all the calculated values
    '''

    total_size = 3 ** len(value_data)
    results = np.zeros(total_size)

    def assign_calc(index, values):
        results[index] = func(*values)

    __iterate_all_values(value_data, error_data, lambda values: assign_calc(values[1], values[0]))
    # print(results)
    return results


def __iterate_all_values(value_data, error_data, on_iterate) -> None:
    '''
    Iterates over all the possible configurations used in mix-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :param on_iterate: Function to be called on every iteration, Arguments is a tuple of (Array, int, Array),
    first argument is an array of all the values calculated in a single iteration and the second argument is
    an integer refering to the iteration number (0 ... N), the third argument is an array of a value configurations
    (0 = default, 1 = min, 2 = max).
    :return:
    '''

    data_size = len(value_data)
    total_size = 3 ** data_size
    iteration_values = np.zeros(data_size)
    value_config = np.zeros(data_size)

    for x in range(0, total_size):
        for y in range(0, data_size):
            value_config[y] = math.floor(x / 3 ** y) % 3

            value_mask = value_config[y]
            value = value_data[y]
            if value_mask == MIN:
                value = value_data[y] - error_data[y]
            elif value_mask == MAX:
                value = value_data[y] + error_data[y]

            iteration_values[y] = value

        # print("x=", x, ": ", value_config)
        on_iterate((iteration_values, x, value_config))