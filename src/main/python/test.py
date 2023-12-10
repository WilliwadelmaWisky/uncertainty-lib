from src.main.python.uncertainty import calc_minmax, calc_standard
import numpy as np

val = np.array([1, 2, 1.5, 0.1])
err = np.array([0.1, 0.2, 0.3, 0.05])

minmax_result, config = calc_minmax(lambda x, y, z, w: (x + y * z) / (y - x * w), val, err)
print("Minmax - value=", minmax_result, ", config=", config)

standard_result = calc_standard(lambda x, y, z, w: (x + y * z) / (y - x * w), 'x y z w', val, err)
print("Standard - value=", standard_result)
