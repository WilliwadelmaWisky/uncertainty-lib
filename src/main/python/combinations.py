import math
import numpy as np


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
            value_mask = config[x, y]
            values[y] = get_masked_value(value_data[y], error_data[y], value_mask)

        on_iterate((x, values, config[x]))


def calculate_all(func, value_data, error_data):
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

    iterate_all(value_data, error_data, lambda values: assign_calc(values[0], values[1]))
    return results
