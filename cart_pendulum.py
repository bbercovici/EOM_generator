import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import EOM_s
from EOMs import DynamicSystem
from EOMs import ExtendedKalman

# Numeric values
m1 = 1
m2 = 0.1
l = 1
l0 = 0.5
k = 0.5
g = 9.81
consts = {'m1': m1,'m2': m2,'k': k,'l': l, 'g': g, 'l0':l0}

# Initial conditions

x0 = 1
theta0 = np.pi/6
xdot0 = 0
thetadot0 = 0
X0 = np.array([x0,theta0,xdot0,thetadot0])

# Minimum set of generalized coordinates
qj = ['x','theta']

# Generalized non-conservative forces
Qj = ['0','0']

# Control variables
controls = []

# TO BE MODIFIED IN AN UPCOMMING RELEASE
# Equilibrium point about which the dynamics must be linearized
equilibrium = ['l','0']

# Kinetic Energy
# (because it is a rather long expression, it can be split in two strings 
# that are later stored in the same list)
T1 = "0.5 * m1 * x_dot(t) ** 2 "
T2 = "0.5 * m2 * ( x_dot(t) ** 2 + (l*theta_dot(t))**2 - 2* x_dot(t) * l * theta_dot(t)* cos(theta(t)))"

# Potential energy
U = "m2 * g  * l * cos(theta(t)) + 0.5 * k * (x(t)-l) * * 2"

 
# Time interval
t0 = 0
tf = 2

# Timestep
dt = 0.01

cart_pendulum = DynamicSystem(qj,[T1,T2],[U],Qj,consts,controls)
cart_pendulum.derive_EOM()
cart_pendulum.lin_dynamics(equilibrium)
cart_pendulum.true_dyn(t0,tf,X0,dt)
cart_pendulum.plot_true_states()

