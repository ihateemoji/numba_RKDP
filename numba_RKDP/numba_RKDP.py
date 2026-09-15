import ctypes as ct
import numba as nb
import numpy as np
import os
import platform

# define the signature of input function
RKDP_sig = nb.types.void(nb.types.double, nb.types.CPointer(nb.types.double), \
        nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double))

def _get_lib_name():
    sysname = platform.system()
    if sysname == "Darwin":
        return "libRKDP.dylib"
    if sysname == "Windows":
        return "libRKDP.dll"
    return "libRKDP.so"

# locate the shared library inside the package
_pkg_root = os.path.dirname(__file__)
_lib_path = os.path.join(_pkg_root, "lib", _get_lib_name())
_lib = ct.CDLL(_lib_path)
# define the signature of C RKDP solver
_lib.RKDP_solver.restype = ct.c_void_p
_lib.RKDP_solver.argtypes = [
    ct.c_void_p,  # x
    ct.c_void_p,  # y (flattened N*M)
    ct.c_void_p,  # f
    ct.c_double,  # eps_rel
    ct.c_int64,   # N
    ct.c_int64,   # M
    ct.c_void_p,  # data ptr
    ct.c_int64    # silent
]
# store C RKDP solver
RKDP_solver = _lib.RKDP_solver

@nb.njit()
def RKDP(func_ptr, x, y0, eps_rel=1e-12, \
                    data=np.array([0.0], np.float64), silent=0):
    """Adaptive Dormand-Prince solver of the system of ODEs.
        Inputs:
            <*func<double, *double, *double, *void>> - pointer to a function
                    corresponding to the RHS of the system of ODEs.
                    first input -> current value of x
                    second input -> pointer to the current solution vector
                    third input -> pointer to the vector to store the RHS in
                    fourth input -> void pointer to any optional data
            <array<float>> - array of the solution domain x
            <array<float>> - array of the initial conditions
        Optional Inputs:
            <float> - target relative tolerance (default 1e-12)
            <array<float>> - array containing any optional data to be sent
                            to the RHS function (default [0.0])
            <int>        - switch for the silent operation
                                0 - silent
                                1 - not silent
                            (default 1)
        Outputs:
            <array<float>> - resulting solution 2D array with first index
                                corresponding to the function and the second
                                index corresponding to the point in x domain"""
    # allocate solution array (column-major layout expected by C)
    total = len(x)*len(y0)
    y = np.empty(total, np.float64)
    # store the initial conditions in the solution array
    y[:len(y0)] = y0
    # call C solver
    RKDP_solver(x.ctypes.data, y.ctypes.data, func_ptr, eps_rel, \
                                len(y0), len(x), data.ctypes.data, silent)
    # reshape to (N, M) then transpose -> (M, N) C-contiguous
    return y.reshape((len(x), len(y0))).T.copy()
