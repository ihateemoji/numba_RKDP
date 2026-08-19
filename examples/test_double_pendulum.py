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
x = np.sin(theta1) + np.sin(theta2)
y = - np.cos(theta1) - np.cos(theta2)
# construct x and y for the first pendulum in units of l
x1 = np.sin(theta1)
y1 = - np.cos(theta1)

# plot the results
fig, ax = plt.subplots()
L = np.max([np.max(np.abs(x)), np.max(np.abs(y))])
ax.set_xlim(-L-0.1, L+0.1)
ax.set_ylim(-L-0.1, L+0.1)
point, = ax.plot([], [], 'o', ms=6)
tail,  = ax.plot([], [], '-', lw=1)
p1,    = ax.plot([], [], '-', lw=5)
p2,    = ax.plot([], [], '-', lw=5)
dt = t[1] - t[0]
interval_ms = 1000 * dt / max(speed, 1e-12)
def init():
    point.set_data([], [])
    tail.set_data([], [])
    p1.set_data([], [])
    p2.set_data([], [])
    return tail, p1, p2, point
def update(k):
    point.set_data([x[k]], [y[k]])
    k0 = max(0, k - tail_len)
    tail.set_data(x[k0:k+1], y[k0:k+1])
    p1.set_data([0, x1[k]], [0, y1[k]])
    p2.set_data([x1[k], x[k]], [y1[k], y[k]])
    return tail, p1, p2, point
ani = FuncAnimation(
    fig, update,
    frames=len(x),
    init_func=init,
    interval=interval_ms,
    blit=True
)
plt.title(r"Double Pendulum Test: $\theta_1 ="+str(round(y0[0],4)) + \
                            r"$, $\theta_2 ="+str(round(y0[2],4))+"$", pad=10)
plt.xlabel("$x / l$")
plt.ylabel("$y / l$")
plt.tight_layout()
plt.show()
