import numpy as np
import numba as nb
from numba_RKDP import RKDP, RKDP_sig
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define the playback parameters
speed = 1.0
tail_len = 400

# define the initial conditions
theta1 = np.random.uniform(-1, 1)*np.pi
dtheta1 = 0.0
theta2 = np.random.uniform(-1, 1)*np.pi
dtheta2 = 0.0
y0 = np.array([theta1, dtheta1, theta2, dtheta2], np.float64)

# define the t domain (in units of sqrt(L / g))
t = np.arange(0, 100, 0.01)

@nb.cfunc(RKDP_sig)
def rhs(x, y_ptr, dydx_ptr, data):
    y = nb.carray(y_ptr, (4,))
    dydx = nb.carray(dydx_ptr, (4,))
    theta1 = y[0]
    dtheta1 = y[1]
    theta2 = y[2]
    dtheta2 = y[3]
    s1 = np.sin(theta1)
    s12 = np.sin(theta1 - theta2)
    c12 = np.cos(theta1 - theta2)
    ddtheta1 = 1/(-16 + 9*c12**2) * \
        (-9*(-2+c12) * s1 + 3*(2*dtheta2**2+3*dtheta1**2*c12)*s12)
    ddtheta2 = -1/(-16 + 9*c12**2) * \
        (3*((-8+9*c12)*s1 + (8*dtheta1**2+3*dtheta2**2*c12)*s12))
    dydx[0] = dtheta1
    dydx[1] = ddtheta1
    dydx[2] = dtheta2
    dydx[3] = ddtheta2

# call solver
sol = RKDP(rhs.address, t, y0)
# unpack
theta1 = sol[0,:]
theta2 = sol[2,:]
# construct x and y in units of l
x = np.sin(theta1) + 1/2*np.sin(theta2)
y = - np.cos(theta1) - 1/2*np.cos(theta2)

# plot the results
fig, ax = plt.subplots()
ax.set_xlim(np.min(x)-0.1, np.max(x)+0.1)
ax.set_ylim(np.min(y)-0.1, np.max(y)+0.1)
point, = ax.plot([], [], 'o', ms=6)
tail,  = ax.plot([], [], '-', lw=1)
dt = t[1] - t[0]
interval_ms = 1000 * dt / max(speed, 1e-12)
def init():
    point.set_data([], [])
    tail.set_data([], [])
    return point, tail
def update(k):
    point.set_data([x[k]], [y[k]])
    k0 = max(0, k - tail_len)
    tail.set_data(x[k0:k+1], y[k0:k+1])
    return point, tail
ani = FuncAnimation(
    fig, update,
    frames=len(x),
    init_func=init,
    interval=interval_ms,
    blit=True
)
plt.title(r"Double Pendulum Test: $\theta_1 ="+str(round(y0[0],4)) + \
                            r"$, $\theta_2 ="+str(round(y[3],4))+"$", pad=10)
plt.xlabel("$x / l$")
plt.ylabel("$y / l$")
plt.tight_layout()
plt.show()
