from src.minmax import calc_minmax_error, calc_minmax_all
import numpy as np

val = np.array([1, 2])
err = np.array([0.1, 0.2])

result, config = calc_minmax_error(lambda x, y: (x + y) / (y - x), val, err)
print(result, ": ", config)

alll = calc_minmax_all(lambda x, y: (x + y) / (y - x), val, err)
print(alll)
