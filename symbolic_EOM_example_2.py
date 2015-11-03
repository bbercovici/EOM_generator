import sympy as sym
import numpy as np
from IPython.display import display
from EOMs import EOM_s
from EOMs import DynamicSystem

# Minimum set of generalized coordinates
qj = ['theta']

# Generalized non-conservative forces
Qj = ['0',]

# Kinetic Energy
T = "0.5 * m * R ** 2 * theta_dot(t) ** 2 "

# Potential energy
U = "0.5 * k * theta(t)**2 - m * g * R * cos( theta(t) )"

spring_pendulum = DynamicSystem(qj,[T],[U],Qj)
spring_pendulum.derive_EOM()
display(spring_pendulum.EOM)
