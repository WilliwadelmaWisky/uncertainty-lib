# Uncertainty Library

A Library for python that helps to calculate uncertanties. 
Currently supports minmax method and standard uncertainty propagation method.

Version 1.0

## Dependencies
- Numpy

## Installation
- Make sure to have dependencies installed to your project
- Download the `uncertainty-lib.zip` from releases
- Unzip and paste to your project

## Examples
NOTE: The import statements may vary depending on where the files a located!

### Standard-method
Calculate an uncertainty of a function `f(x, y) = x^2 * y^2` at `{ x = (1.0 ± 0.2), y = (2.0 ± 0.3) }` using a standard uncertainty propagation method.

```python
from uncertainty import standard
from numpy import array

f = lambda x, y: x**2 * y**2
point = array([1.0, 2.0])
point_err = array([0.2, 0.3])

err = standard(f, point, point_err)
```

The `err` variable now has the calculated uncertainty `2`.

### Minmax-method
Calculate an uncertainty of a function `f(x, y) = x^2 * y^2` at `{ x = (1.0 ± 0.2), y = (2.0 ± 0.3) }` using a minmax (worst-case-scenario) uncertainty propagation method. 

```python
from uncertainty import minmax
from numpy import array

f = lambda x, y: x**2 * y**2
point = array([1.0, 2.0])
point_err = array([0.2, 0.3])

err = minmax(f, point, point_err)
```

The `err` variable now has the calculated uncertainty `3.617`
