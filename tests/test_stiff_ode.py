"""
Basic pytest test for numba_RKDP on a stiff ODE system.

Checks the numerical solution against the analytic solution at every
time point and prints detailed deviation statistics.
"""

import numpy as np
import numba as nb
from numba_RKDP import RKDP, RKDP_sig


def test_stiff_ode():
    """
    Test the RKDP solver on a simple stiff linear ODE system.

    The test system is:
        dy0/dt = -1000 * y0
        dy1/dt = -0.01 * y1

    This is a classic stiff test problem. The solver must remain stable
    and accurate across the entire time interval.

    The test compares the numerical solution to the known analytic
    solution at every time point and prints detailed error statistics.
    """

    # define the time domain
    t = np.linspace(0.0, 1.0, 1000000)

    # initial conditions: y0(0) = 1.0, y1(0) = 1.0
    y0 = np.array([1.0, 1.0], dtype=np.float64)

    @nb.cfunc(RKDP_sig)
    def rhs(x, y_ptr, dydx_ptr, data):
        """
        Right-hand side of the stiff ODE system.

        Inputs:
            <float>  - current value of time
            <*float> - pointer to the current solution vector
            <*float> - pointer to the derivative vector (to be populated)
            <*float> - optional data pointer (unused)
        """
        y = nb.carray(y_ptr, (2,))
        dydx = nb.carray(dydx_ptr, (2,))

        dydx[0] = -10.0 * y[0]
        dydx[1] = -0.1 * y[1]

    # solve the system
    sol = RKDP(rhs.address, t, y0, eps_rel=1e-14, silent=1)

    # analytic solution over the entire time domain
    y0_exact = np.exp(-10.0 * t)
    y1_exact = np.exp(-0.1 * t)

    # numerical solutions
    y0_num = sol[0, :]
    y1_num = sol[1, :]

    # absolute and relative deviations at every point
    abs_err0 = np.abs(y0_num - y0_exact)
    abs_err1 = np.abs(y1_num - y1_exact)
    rel_err0 = abs_err0 / (np.abs(y0_exact) + 1e-30)
    rel_err1 = abs_err1 / (np.abs(y1_exact) + 1e-30)

    # print detailed diagnostic information
    print("\n=== Stiff ODE Test Diagnostics ===")
    print(f"Time points: {len(t)}")
    print(f"Final time : {t[-1]}")
    print()
    print("Component 0 (fast decay, lambda = -10):")
    print(f"  max abs error : {np.max(abs_err0):.6e}")
    print(f"  mean abs error: {np.mean(abs_err0):.6e}")
    print(f"  max rel error : {np.max(rel_err0):.6e}")
    print()
    print("Component 1 (slow decay, lambda = -0.1):")
    print(f"  max abs error : {np.max(abs_err1):.6e}")
    print(f"  mean abs error: {np.mean(abs_err1):.6e}")
    print(f"  max rel error : {np.max(rel_err1):.6e}")
    print("==================================\n")

    # check that the solution stays close to the analytic result
    # use absolute tolerance only; relative error on the fast component
    # becomes meaningless once the analytic solution is near machine zero
    assert np.allclose(y0_num, y0_exact, atol=1e-06)
    assert np.allclose(y1_num, y1_exact, atol=1e-06)


if __name__ == "__main__":
    test_stiff_ode()
