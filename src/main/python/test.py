from src.main.python.minmax import calculate
from src.main.python.combinations import calculate_all
import numpy as np

val = np.array([1, 2, 1.5, 0.1])
err = np.array([0.1, 0.2, 0.3, 0.05])

result, config = calculate(lambda x, y, z, w: (x + y * z) / (y - x * w), val, err)
print(result, ": ", config)

print(calculate_all(lambda x, y, z, w: (x + y * z) / (y - x * w), val, err))
