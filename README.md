# numba_RKDP

Adaptive Dormand–Prince (RKDP) ODE solver exposed to Python/Numba via a small C shared library.

This repository includes a compact C implementation of an adaptive Dormand–Prince (5(4)) integrator and a tiny Python wrapper, allowing you to call the compiled solver from Numba-compiled callbacks (via cfunc). It’s designed for simple, fast, adaptive ODE solves that can be invoked from Numba code.

Installation
Requirements:
- Python 3.6+
- numpy
- numba
- gcc (C99) with OpenMP support (or another compiler that supports -fopenmp)

Build and install locally:
```
# compile the shared library
make

# optional: install package in editable mode
pip install -e .
```

Or:
```
pip install .
```

Example pattern:
```
import numpy as np
import numba as nb
from numba_RKDP import RKDP, RKDP_sig

@nb.cfunc(RKDP_sig)
def rhs(x, y_ptr, dydx_ptr, data):
    y = nb.carray(y_ptr, (4,))
    dydx = nb.carray(dydx_ptr, (4,))
    # fill dydx[...] from y[...]

y0 = np.array([theta1, dtheta1, theta2, dtheta2], np.float64)
t = np.arange(0, 100, 0.01)

sol = RKDP(rhs.address, t, y0)
# sol shape: (num_vars, num_points)
```

Notes & troubleshooting
- If Python cannot find the shared library, ensure you ran `make` at the repo root (setup.py runs it automatically on install).
- The wrapper locates the .so relative to the package directory; installing the package with pip or using editable mode makes imports straightforward.
- The C code uses OpenMP pragmas; the Makefile compiles and links with `-fopenmp`. If your compiler lacks OpenMP, remove the flags or install a compiler that supports OpenMP.
- The example uses matplotlib for animation; matplotlib is not required for the core solver.
