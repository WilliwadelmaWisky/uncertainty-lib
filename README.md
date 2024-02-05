# Uncertainty Library

A Library for python that helps to calculate uncertanties. 
Currently supports minmax method and standard uncertainty propagation method.

Work in Progress

## Dependencies
- Numpy

## Installation
- Make sure to have dependencies installed to your project
- Bring the contents of src/ to your project
	1. Download the zip, unzip and copy the contents of src/ to your project or
	2. Clone the repo and copy the contents of src/ to your project or
	3. Copy/Paste the scripts

## Examples
NOTE: The import statements may vary dependeding on where the files a located!

### Standard-method
Calculate an uncertainty of a function `f(x, y) = x^2 * y^2` at `{ x = (1.0 ± 0.2), y = (2 ± 0.3) }` using a standard uncertainty propagation method.

```python
from uncertainty import standard
from numpy import array

f = lambda x, y: x**2 * y**2
point = array([1.0, 0.2])
point_err = array([0.2, 0.3])

err = standard(f, point, point_err)
```

The `err` variable now has the calculated uncertainty `~3.617`.

### Minmax-method
Calculate an uncertainty of a function `f(x, y) = x^2 * y^2` at `{ x = (1.0 ± 0.2), y = (2 ± 0.3) }` using a minmax uncertainty propagation method. 

```python
from uncertainty import minmax
from numpy import array

f = lambda x, y: x**2 * y**2
point = array([1.0, 0.2])
point_err = array([0.2, 0.3])

err = minmax(f, point, point_err)
```

The `err` variable now has the calculated uncertainty `2`
