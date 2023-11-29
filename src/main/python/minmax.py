import numpy as np
from src.main.python.combinations import calculate_all, all_combinations


def calculate(func, value_data, error_data):
    '''
    Calculates an uncertainty using a min-max method, calculation is expensive because it uses bruteforce approach
    :param func: Function to apply min-max method
    :param value_data: All the variables required in a function arguments
    :param error_data: All the errors to variables, has to be of same size as value_data array
    :return: Tuple of (float, Array), first value is the calculated error and the second value
    is an array of value configurations (0 = default, 1 = min, 2 = max). Index of the array indicates
    position of the argument
    '''

    ref = func(*value_data)
    all_values = calculate_all(func, value_data, error_data)
    all_errors = np.abs(all_values - ref)
    err = np.max(all_errors)

    max_index = np.where(all_errors == err)[0][0]
    all_configs = all_combinations(len(value_data))
    config = all_configs[max_index]

    return err, config
