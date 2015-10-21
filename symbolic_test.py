import sympy as sym
import numpy as np

theta, theta_dot, x,x_dot ,theta_ddot, x_ddot= sym.symbols('theta theta_dot  x x_dot theta_ddot x_ddot')
l, t , m,g = sym.symbols('l t m g')

# Generalized coordinates
gen_cords = {'0':theta(t)}
gen_vels = {'0':theta_dot(t)}
gen_accs = {'0':theta_ddot}

if(len(gen_cords)!=len(gen_vels) or len(gen_cords)!=len(gen_accs) or len(gen_vels)!=len(gen_accs)):
	raise NameError('there should be an equal number of generalized coordinates, velocities and accelerations')



# Kinetic energy
T = 0.5 * m * l ** 2 * theta_dot(t) ** 2 

# Potential energy
V = - m * g * l * sym.cos(theta(t))

# Lagrangian
L = T - V

# Array of equation components
L1 = sym.Matrix( np.zeros([len(gen_cords),1]))
L2 = sym.Matrix( np.zeros([len(gen_cords),1]))

# Step 1: partial w.r to generalized velocities
for gen_vel in gen_vels:
	L1[int(gen_vel)] = L.diff(gen_vels[gen_vel])

# Step 2: Taking the time derivative of the previously computed partials
for gen_vel in gen_vels:
	L1[int(gen_vel)] = L1[int(gen_vel)].diff(t)

# Step 3: Computing the partial derivative w.r to the generalized coordinates
for gen_cord in gen_cords:
	L2[int(gen_cord)] = L.diff(gen_cords[gen_cord])
		
# Step 4: Summing the previously computed partials to obtain the LHS of the unsorted EOMs
LHS = L1 - L2

# Step 5: Substituting the non-evaluated second-order derivative with expressions that can be solved for in the next step
# and removing the time dependance in the generalized coordinates/velocities
for i in range (len(gen_cords)):
	for gen_acc in gen_accs:
		LHS[i] = LHS[i].subs(sym.diff(gen_vels[gen_acc]),gen_accs[gen_acc])

# Step 6: Solving the equations for the second-order time derivative of the generalized coordinates
solver_args = [LHS] + gen_accs.values()
X_ddot =  sym.solve(*solver_args)
print X_ddot
