import sympy as sym
import numpy as np

def cart_dot(X):
	ex, ey, ez = sym.symbols('ex ey ez')
	X = X.expand(basic=True)
	X = sym.collect(X,[ex**2,ey**2,ez**2,ex*ey,ey*ez,ez*ex])
	return X.subs([(ex**2,1),(ey**2,1),(ez**2,1),(ex*ey,0),(ey*ez,0),
	(ez*ex,0),(ez*ey,0),(ex*ez,0),(ey*ez,0)])

theta, theta_dot, theta_ddot = sym.symbols('theta theta_dot  theta_ddot ', real = True)
x,x_dot ,x_ddot = sym.symbols('x x_dot x_ddot', real= True)
y,y_dot ,y_ddot = sym.symbols('y y_dot y_ddot', real= True)
z,z_dot ,z_ddot = sym.symbols('z z_dot z_ddot', real= True)

dx,dy,dz = sym.symbols('dx dy dz')
l, t , m,g = sym.symbols('l t m g', real = True)
ex, ey, ez = sym.symbols('ex ey ez')
er = sym.cos(theta(t)) * ex + sym.sin(theta(t)) * ey
etheta = - sym.sin(theta(t)) * ex + sym.cos(theta(t)) * ey

# Generalized coordinates
gen_cords = {'0':theta(t)}
gen_vels = {'0':theta_dot(t)}
gen_accs = {'0':theta_ddot}
if(len(gen_cords)!=len(gen_vels) or len(gen_cords)!=len(gen_accs) or len(gen_vels)!=len(gen_accs)):
	raise NameError('there should be an equal number of generalized coordinates, velocities and accelerations')

# Inertial positions
R1 = l * er 
R = sym.Matrix([R1])

# Particle masses
M = sym.Matrix([m])

# Inertial velocities
V = sym.Matrix(np.zeros(len(R)))
for i in range(len(R)):
	V[i] = sym.diff(R[i],t)


# Kinetic energy
T = 0
for i in range(len(R)):
	T = T + 0.5 * M[i] * V[i]**2
T = T.expand(basic=True)
T = cart_dot(T)

for gen_vel in gen_vels:
	T = sym.simplify(T.subs(sym.diff(gen_cords[gen_vel],t),
		gen_vels[gen_vel]))

# Potential energy
U = 0
F = -m*g*ez

# WARNING. NOT SUITED TO CENTRAL FORCES SUCH AS RADIAL GRAVITY
Ep = - cart_dot(F*(x*ex + y*ey + z*ez))



# # # Kinetic energy
# # T = 0.5 * m * l ** 2 * theta_dot(t) ** 2 

# # Potential energy
# V = m * g * l * sym.cos(theta(t))

# # Lagrangian
# L = T - V

# # Array of equation components
# L1 = sym.Matrix( np.zeros([len(gen_cords),1]))
# L2 = sym.Matrix( np.zeros([len(gen_cords),1]))

# # Step 1: partial w.r to generalized velocities
# for gen_vel in gen_vels:
# 	L1[int(gen_vel)] = L.diff(gen_vels[gen_vel])

# # Step 2: Taking the time derivative of the previously computed partials
# for gen_vel in gen_vels:
# 	L1[int(gen_vel)] = L1[int(gen_vel)].diff(t)

# # Step 3: Computing the partial derivative w.r to the generalized coordinates
# for gen_cord in gen_cords:
# 	L2[int(gen_cord)] = L.diff(gen_cords[gen_cord])
		
# # Step 4: Summing the previously computed partials to obtain the LHS of the unsorted EOMs
# LHS = L1 - L2

# # Step 5: Substituting the non-evaluated second-order derivative with expressions that can be solved for in the next step
# for i in range (len(gen_cords)):
# 	for gen_acc in gen_accs:
# 		LHS[i] = LHS[i].subs(sym.diff(gen_vels[gen_acc]),gen_accs[gen_acc])

# # Step 6: Solving the equations for the second-order time derivative of the generalized coordinates
# solver_args = [LHS] + gen_accs.values()
# X_ddot =  sym.solve(*solver_args)
# print X_ddot

