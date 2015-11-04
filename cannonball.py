import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import EOM_s
from EOMs import DynamicSystem

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

# Equilibrium point about which the dynamics must be linearized
equilibrium = ['l','0']

# Kinetic Energy
# (because it is a rather long expression, it can be split in two strings 
# that are later stored in the same list)
T = "0.5 * m * (x_dot(t) ** 2 + y_dot(t) ** 2) "

# Potential energy
U = "m * g  * y(t)"

# Initial state
X0 = np.array([0,0,10,10])
 
# Time interval
t0 = 0
tf = 10

# Timestep
dt = 1


cannonball = DynamicSystem(qj,[T],[U],Qj,consts,controls)
cannonball.derive_EOM()
cannonball.lin_dynamics(equilibrium)
cannonball.true_dyn(t0,tf,X0,dt)
cannonball.plot_true_states()

