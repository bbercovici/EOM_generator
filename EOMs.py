import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

        
class DynamicSystem:
    """Defines the DynamicSystem Class"""
    def __init__(self, g_cords, kin_e , pot_e, gen_forces,
        consts,controls):
        self.g_cords = g_cords
        self.kin_e = '+'.join(kin_e)
        self.pot_e = '+'.join(pot_e)
        self.gen_forces = gen_forces
        self.consts = consts
        self.controls = controls
    def derive_EOM(self):
        [self.EOM,self.ns] = EOM_s(self.g_cords,self.kin_e,self.pot_e,
            self.gen_forces,self.consts,self.controls)
    def lin_dynamics(self,equilibrium):
        [self.F,self.G] = lin_dynamics(self.g_cords,
            self.EOM,self.consts,self.controls,equilibrium)
    def X_dot(self):
        self.Xdot = X_dot(self.g_cords, self.EOM,self.controls,
            self.consts)
    def true_dyn(self,t0,tf,X0,dt):
        self.X_dot()
        [self.T,self.X] = true_dyn(self.Xdot,t0,tf,X0,dt)
    def plot_true_states(self):
        plot_true_states(self.T,self.X,self.g_cords);




def EOM_s(g_cords, kin_e , pot_e, gen_forces,consts,controls):
    '''
    Returns a symbolic vector representing the EOM of a dynamical system.
    This dynamical system
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
    consts : (dictionnary) constant values
    controls : (list of strings) control variables

    Outputs:
    --------
    EOM: (dictionnary) symbolic equations of motion expressed in terms of the generalized
        coordinates, their time derivatives and problem constants
    ns : (dictionnary) symbolic variables already defined

    '''

    # Some key symbolic variables are defined here
    t = sym.symbols('t', real = True)
    gen_cords = sym.Matrix(np.zeros(len(g_cords)))
    gen_cords_implementation = sym.Matrix(np.zeros(len(g_cords)))

    gen_vels = sym.Matrix(np.zeros(len(g_cords)))
    gen_vels_implementation = sym.Matrix(np.zeros(len(g_cords)))

    gen_accs = sym.Matrix(np.zeros(len(g_cords)))

    for i in range(len(gen_cords)):
        gen_cords[i] = sym.symbols(g_cords[i], real = True)(t)
        gen_cords_implementation[i] = sym.symbols(g_cords[i], real = True)
        gen_vels[i] = sym.symbols(g_cords[i]+'_dot', real = True)(t)
        gen_vels_implementation[i] = sym.symbols(g_cords[i]+'_dot', real = True)
        gen_accs[i] = sym.symbols(g_cords[i]+'_ddot', real = True)
    

    # Dictionnary of local symbols already defined
    ns = {}
    for sym_cord in gen_cords:
        ns[str(sym_cord)] = sym.Symbol(str(sym_cord), real=True)
    for sym_vel in gen_vels:
        ns[str(sym_vel)] = sym.Symbol(str(sym_vel), real=True)
    for control in controls:
        ns[str(control)] = sym.Symbol(str(control), real=True)
    
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

    # Step 7: the computed EOMs are reformatted so as to appear in state
    # space form (i.e in a "ready to implement form")
    EOM_implementation = sym.Matrix(np.zeros(2*len(gen_cords)))
    
    for i in range(len(gen_cords)):
        EOM_implementation[i] = gen_vels_implementation[i]
        EOM_implementation[i+2] = EOM[gen_accs[i]]
    for i in range(len(gen_cords)):
        EOM_implementation = EOM_implementation.subs(gen_cords[i],
            gen_cords_implementation[i])
        EOM_implementation = EOM_implementation.subs(gen_vels[i],
            gen_vels_implementation[i])
    return [EOM_implementation,ns]

def lin_dynamics(g_cords,EOM,consts,controls,equilibrium):
    '''
    Return the function handles to the state-space matrix F and the state-control matrix G in 
    Xdot = Fx + Gu
    Parameters:
    -----------
    g_cords: (list of strings) generalized coordinates (Ex: g_cords == ['x','theta'])
    EOM : (symbolic matrix) equations of motion in a ready to implement form
    consts : (dictionnary) constant values
    controls : (list of strings) control variables
    equilibrium : (list of strings) equilibrium point
    Returns:
    -----------
    F : (function handle) state-space matrix 
    G : (function handle) control state matrix 
    '''
    state = sym.Matrix(np.zeros(2*len(g_cords)))
    control = sym.Matrix(np.zeros(len(controls)))

    for i in range(len(g_cords)):
        state[i] = sym.symbols(g_cords[i], real = True)
        state[i+len(g_cords)] = sym.symbols(g_cords[i] + '_dot', real = True)

    for i in range(len(controls)):
        control[i] = sym.symbols(controls[i], real = True)

    # Symbolic Jacobian matrices
    Fs = EOM.jacobian(state)    
    Gs = EOM.jacobian(control)
    
    state_control_time = sym.Matrix(np.zeros(2*len(g_cords) + len(controls)+1))
    t = sym.symbols('t', real = True)
    state_control_time[0] = t
    for i in range(len(g_cords)):
        state_control_time[i + 1] = sym.symbols(g_cords[i], real = True)
        state_control_time[i + len(g_cords)+ 1] = sym.symbols(g_cords[i]+'_dot', real = True)
    for i in range(len(controls)):
        state_control_time[2*len(g_cords)+ i + 1] = sym.symbols(controls[i], real = True)


    F = sym.lambdify(state_control_time,Fs.subs(consts), modules='numpy')
    G = sym.lambdify(state_control_time,Gs.subs(consts), modules='numpy')


    return [F,G]

def X_dot(g_cords,EOM,controls,consts):
    """
    Returns a function handle to the non-linear state rates
    Parameters:
    -----------
    g_cords: (list of strings) generalized coordinates (Ex: g_cords == ['x','theta'])
    EOM : (symbolic matrix) equations of motion in a ready to implement form
    consts : (dictionnary) constant values
    controls : (list of strings) control variables
    Returns:
    ---------
    state_rates : (function handle) function handle to the state rates
    """
    state_control_time = sym.Matrix(np.zeros(2*len(g_cords) + len(controls)+1))
    t = sym.symbols('t', real = True)
    state_control_time[0] = t
    for i in range(len(g_cords)):
        state_control_time[i + 1] = sym.symbols(g_cords[i], real = True)
        state_control_time[i + len(g_cords)+ 1] = sym.symbols(g_cords[i]+'_dot', real = True)
    for i in range(len(controls)):
        state_control_time[2*len(g_cords)+ i + 1] = sym.symbols(controls[i], real = True)
    state_rates = sym.lambdify(state_control_time,EOM.subs(consts), modules='numpy')
    return state_rates


def dxdt_interface(X,t , dxdt):
    """
    Provides an interface between odeint and dxdt
    Parameters :
    ------------
    X : (n-by-1 np array) state
    t : time
    dxdt : (function handle) time derivative of the true (n-by-1) state vector
    Returns:
    --------
    (n-by-1 np.array) time derivative of the components of the  state 
    """
    xdot = np.array(dxdt(*list(np.append(t,X))))
    xdot = xdot.reshape([len(xdot),])
    return xdot


def true_dyn(dXdt,t0,tf,X0,dt):
    """
    Propagates the dynamics of the true state forward
    using a 4-th order Runge Kutta
    Parameters:
    -----------
    dXdt: (function handle) time rates of the true state
    t0 : initial time
    tf : final time
    X0 : (n-by-1) initial state
    dt : time step
    Returns:
    -----------
    T : (k-by-1 np.array) time histories
    X : (n-by-k np.array) state histories
    """
    T = np.linspace(t0,tf,round(1/dt))

    # X = odeint(dxdt_interface , X0.reshape([len(X0),]), [t0,tf], args = (dXdt,),
    #     rtol = 3e-14 , atol = 1e-16)
    X = odeint(dxdt_interface , X0.reshape([len(X0),]), T,
        args = (dXdt,),rtol = 3e-14 , atol = 1e-16)

    # T = np.linspace(t0,tf,round((tf-t0)/dt))
    # X = np.zeros([len(X0),len(T)])
    # X[:,0] = X0
    # # Non-linear dynamics are propagated forward in time
    # for i in range(len(T)-1):
    #     k1 = np.squeeze(dXdt(*list(np.append(T[i],X[:,i]))))
    #     k2 = np.squeeze(dXdt(*list(np.append(T[i] + dt/2 , X[:,i] + dt/2 * k1))))
    #     k3 = np.squeeze(dXdt(*list(np.append(T[i] + dt/2 , X[:,i]+ dt/2 * k2))))
    #     k4 = np.squeeze(dXdt(*list(np.append(T[i] + dt , X[:,i] + dt * k3 ))))
    #     X[:, i + 1 ] = X[:, i ] + dt/6. * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return [T,X]



def state_obs_mat(ns,g_cords,state_obs):
    """
    Returns a function handle to the state observation matrix Htilde
    Parameters :
    ------------
    ns : (dictionnary) local symbols
    g_cords : (list of strings) generalized coordinates
    state_obs : (list of strings) obsevation equations
    Returns:
    --------
    (function handle) state observation matrix 
    """
    state = sym.Matrix(np.zeros(2*len(g_cords)))

    for i in range(len(g_cords)):
        state[i] = sym.symbols(g_cords[i], real = True)
        state[i+len(g_cords)] = sym.symbols(g_cords[i] + '_dot', real = True)

    state_obs_s = sym.sympify(state_obs, locals = ns)

    Htilde_s = state_obs_s.jacobian(state)

    t = sym.symbols('t', real = True)
    state_time[0] = t
    for i in range(len(g_cords)):
        state_time[i + 1] = sym.symbols(g_cords[i], real = True)
        state_time[i + len(g_cords)+ 1] = sym.symbols(g_cords[i]+'_dot', real = True)
    Htilde = sym.lambdify(state_time,Htilde_s, modules='numpy')

    return Htilde


def plot_true_states(T,X,g_cords):
    """
    Plots the true states' history
    Parameters:
    -----------
    T: (k-by-1 np.array) time history
    X : (n-by-k np.array) state time histories
    g_cords: (list of strings) generalized coordinates (Ex: g_cords == ['x','theta'])
    """
    for i in range(len(g_cords)):
        plt.subplot(2*len(g_cords), 1, i*len(g_cords)+1)
        plt.plot(T, X[i,:])
        plt.ticklabel_format(useOffset=False)
        plt.title('$' + g_cords[i] + '$')
        plt.ylabel('$' + g_cords[i] + '$')
    
        plt.subplot(2*len(g_cords), 1, i*len(g_cords)+2)
        plt.plot(T, X[i+len(g_cords),:])
        plt.ticklabel_format(useOffset=False)
        plt.title('$\dot{'+ g_cords[i] + "}$")
        plt.ylabel('$\dot{'+ g_cords[i] + "}$")
    plt.xlabel('Time (s)')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5,forward=True)
    plt.savefig('true_states.pdf', bbox_inches='tight')


