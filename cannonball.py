import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import EOM_s
from EOMs import DynamicSystem
from EKFs import ExtendedKalman

# Numeric values
m = 1
g = 9.81
consts = {'m': m, 'g': g}

# Minimum set of generalized coordinates
qj = ['x','y']

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
T = "0.5 * m * (x_dot(t) ** 2 + y_dot(t) ** 2) "
T = [T]

# Potential energy
U = "m * g  * y(t)"
U = [U]

# Initial state
X0 = np.array([0,0,10,10])
 
# Time interval
t0 = 0
tf = 10

# Timestep
dt = 1


cannonball = DynamicSystem(qj,T,U,Qj,consts,controls)
cannonball.derive_EOM()
cannonball.lin_dynamics(equilibrium)
cannonball.true_dyn(t0,tf,X0,dt)
# cannonball.plot_true_states()


## EKF Setup
state_obs = []
R = np.eye(2)
X0_bar = X0
P0 = np.eye(4)

EKF = ExtendedKalman(X0_bar,P0,R,cannonball,state_obs)
EKF.compute_estimate()

