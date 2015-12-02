import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import DynamicSystem
from EKFs import ExtendedKalman
from CKFs import ClassicalKalman

# Numeric values
m = 1
g = 9.81
c = 0.01
prob_consts = {'m': m, 'g': g, 'c':c}

# Minimum set of generalized coordinates
qj = ['x','y']

# Generalized non-conservative forces
Qj = ['- c * x_dot(t)','-c * y_dot(t)']

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
 
# Dynamics time interval
t0 = 0
tf = 0.01

# Timestep
dt = 0.001


cannonball = DynamicSystem(qj,T,U,Qj,prob_consts,controls)
cannonball.derive_EOM()
cannonball.lin_dynamics(equilibrium)
cannonball.true_dyn(t0,tf,X0,dt)
# cannonball.plot_true_states()


## EKF Setup
# Observation relations
xs = 1
ys = 1
EKF_consts = {'xs' : xs,'ys' : ys}

obs = 4

P0 = np.diag([1,1,1,1])
X0_bar = np.random.multivariate_normal(X0, P0)

EKF = ExtendedKalman(X0_bar,P0,cannonball,obs,EKF_consts)
EKF.compute_estimate()
EKF.plot_estimate()
EKF.plot_residuals()

CKF = ClassicalKalman(X0_bar,P0,cannonball,obs,EKF_consts)
CKF.compute_estimate()
CKF.plot_estimate()
CKF.plot_residuals()

