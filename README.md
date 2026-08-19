# numba_RKDP

`numba_RKDP` is an Adaptive Dormand–Prince (RKDP) ODE solver exposed to Python/Numba via a small C shared library.

## Installation
Requirements:
- Python 3.6+
- numpy
- numba
- gcc (C99) with OpenMP support

The package can be easily built and installed via pip:
```
pip install .
```
## Example usage
Let's solve the following simple system of ODEs:
```
y' = 2 z - 3 y + exp(t),
z' = - y + exp(2*t),
```
subject to initial conditions
```
y(0) = -1/3,
z(0) = 0.
```
We start by importing required packages into the workspace and defining the time-domain for numerical integration:
```
import numpy as np
import numba as nb
from numba_RKDP import RKDP, RKDP_sig

t = np.linspace(0, 1, num=500)
```
To proceed, we need to package our system into a vector; the simplest way to achieve this is to define the vector (y, z) allowing us to define our initial conditions as
```
y0 = np.array([-1/3, 0], np.float64)
```
With this, we can code up the right-hand side of our system of ODEs as a procedure with a signature expected by the RKDP solver 
```
@nb.cfunc(RKDP_sig)
def rhs(t, y_ptr, dydx_ptr, data):
  y_in = nb.carray(y_ptr, (2,))
  dydx_out = nb.carray(dydx_ptr, (2,))
  y = y_in[0]
  z = y_in[1]
  dydx_out[0] = 2*z - 3*y + np.exp(t)
  dydx_out[1] = - y + np.exp(2*t)
```
Note that our procedure uses the current solution vector to populate the time derivative vector. Now we have everything ready to solve the system of equations with
```
sol = RKDP(rhs.address, t, y0)
```
and unpack the solutions
```
y = sol[0,:]
z = sol[1,:]
```
For additional examples, see "examples" folder in the repository.
