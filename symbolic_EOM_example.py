import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import EOM_s
from EOMs import DynamicSystem

# Minimum set of generalized coordinates
qj = ['x','theta']

# Generalized non-conservative forces
Qj = ['u','0']

# Kinetic Energy
T1 = "0.5 * m1 * x_dot(t) ** 2 "
T2 = "0.5 * m2 *( x_dot(t) ** 2 + (l*theta_dot(t))**2 - 2* x_dot(t) * l * theta_dot(t)* cos(theta(t)))"

# Potential energy
U = "m2 * g  * l * cos(theta(t))+0.5*k*(x(t)-l)**2"

cart_pendulum = DynamicSystem(qj,[T1,T2],[U],Qj)
cart_pendulum.derive_EOM()
display(cart_pendulum.EOM)
