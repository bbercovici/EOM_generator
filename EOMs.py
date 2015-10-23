import sympy as sym
import numpy as np


class DynamicSystem:
	"""Defines the DynamicSystem Class"""
	def __init__(self, g_cords, kin_e , pot_e, gen_forces):
		self.g_cords = g_cords
		self.kin_e = '+'.join(kin_e)
		self.pot_e = '+'.join(pot_e)
		self.gen_forces = gen_forces
	def derive_EOM(self):
		self.EOM = EOM_s(self.g_cords,self.kin_e,self.pot_e,
			self.gen_forces)

def EOM_s(g_cords, kin_e , pot_e, gen_forces):
	'''
	Returns a dictionnary storing the symbolic expressions of the 
	second-order time derivatives of a prescribed set of generalized
	coordinates. In other words, this function returns the symbolic 
	equations of motions for a dynamical system. This dynamical system
	is fully defined by a set of MINIMUM generalized coordinates, its 
	kinetic energy and potential energy, the latter being formulated in terms
	of the prescribed generalized coordinates. Lagrange's equations are used to solve for
	the dynamical unknowns
	Parameters:
	-----------
	g_cords: (list of strings) generalized coordinates (Ex: g_cords == ['x','theta'])
	kin_e: (string) kinetic energy expressed in terms of 
		the generalized coordinates, their time derivatives and 
		problem constants. Note that the time dependence must be 
		explicit (Ex: kin_e == '0.5 * m * x_dot(t)**2')
	pot_e: (string) potential energy expressed in terms of 
		the generalized coordinates and 
		problem constants. Note that the time dependence must also be 
		explicit (Ex: pot_e == 'm * g * x(t)')
	gen_forces : (list of strings) generalized forces entered in the same
		order as the generalized coordinates
	Outputs:
	--------
	EOM: (dictionnary) symbolic equations of motion expressed in terms of the generalized
		coordinates, their time derivatives and problem constants

	'''

	# Some key symbolic variables are defined here
	t = sym.symbols('t', real = True)
	gen_cords = sym.Matrix(np.zeros(len(g_cords)))
	gen_vels = sym.Matrix(np.zeros(len(g_cords)))
	gen_accs = sym.Matrix(np.zeros(len(g_cords)))

	for i in range(len(gen_cords)):
		gen_cords[i] = sym.symbols(g_cords[i], real = True)(t)
		gen_vels[i] = sym.symbols(g_cords[i]+'_dot', real = True)(t)
		gen_accs[i] = sym.symbols(g_cords[i]+'_ddot', real = True)
	

	# Dictionnary of local symbols already defined
	ns = {}
	for sym_cord in gen_cords:
		ns[str(sym_cord)] = sym.Symbol(str(sym_cord), real=True)
	for sym_vel in gen_vels:
		ns[str(sym_vel)] = sym.Symbol(str(sym_vel), real=True)
	
	ns['t'] = t

	# The kinetic energy and the potential energy are "sympified"
	# using the symbols that have already been defined
	T = sym.sympify(kin_e, locals = ns)
	U = sym.sympify(pot_e, locals = ns)
	
	# Lagrangian
	L = T - U

	# Array of equation components
	L1 = sym.Matrix( np.zeros([len(gen_cords),1]))
	L2 = sym.Matrix( np.zeros([len(gen_cords),1]))

	# Step 1: partial w.r to generalized velocities
	for i in range(len(gen_vels)):
		L1[i] = L.diff(gen_vels[i])
	
	# Step 2: Taking the time derivative of the previously computed partials
	for i in range(len(gen_vels)):
		L1[i] = L1[i].diff(t)

	# Step 3: Computing the partial derivative w.r to the generalized coordinates
	for i in range(len(gen_cords)):
		L2[i] = L.diff(gen_cords[i])
	
	# Step 4: Summing the previously computed partials to obtain the LHS of the unsorted EOMs
	LHS = L1 - L2

	# Step 5: Substituting the non-evaluated first and second-order derivative with 
	# expressions that can be solved for in the next step
	for i in range (len(gen_cords)):
		for k in range (len(gen_accs)):
			LHS[i] = LHS[i].subs(sym.diff(gen_vels[k]),gen_accs[k])
			LHS[i] = LHS[i].subs(sym.diff(gen_cords[k]),gen_vels[k])

	# Step 6: Solving the equations for the second-order 
	# time derivative of the generalized coordinates after substracting the 
	# generalized forces to the left hand side
	RHS = sym.Matrix(sym.sympify(gen_forces, locals = ns) )

	solver_args = [sym.simplify(sym.expand(LHS-RHS))] + list(gen_accs)
	
	EOM = sym.solve(*solver_args)
	return EOM
