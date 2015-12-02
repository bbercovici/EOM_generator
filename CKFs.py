import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def identity(t,*args):
    return np.array(args).reshape(len(args),1)

def time_state(t,X):
    return [t] + list(X)

class ClassicalKalman:
    """Defines the ClassicalKalman Class"""
    def __init__(self, X0,P0,dyn_sys,obs,consts):
        self.X0 = X0
        self.P0 = P0
        self.dyn_sys = dyn_sys
        self.A = dyn_sys.F
        self.N_obs = obs

        range_m = ['((x(t)-xs)**2 + (y(t)-ys)**2)**(0.5)']

        if(obs == 2):
            self.R =  np.diag([1**2,0.1**2])
        elif obs == 4 :
            self.R = np.diag([1**2,1**2,0.1**2,0.1**2])
        else :
            raise ValueError('obs == 2 or 4')

        if obs == 4:
            self.Htilde = lambda t, *args : np.eye(len(args))
            self.G = identity

        else:
            self.range_s = sym.Matrix(sym.sympify(range_m,locals = dyn_sys.ns)).subs(consts)
            self.range_rate_s = derive_range_rate_s(self.range_s,dyn_sys.g_cords,dyn_sys.ns)
            self.state_obs_eq = sym.Matrix(np.zeros(2))
            self.state_obs_eq[0] = self.range_s
            self.state_obs_eq[1] = self.range_rate_s
            [self.Htilde,self.G] = state_obs_funs(dyn_sys.g_cords,self.state_obs_eq)


        self.T_obs = dyn_sys.T
        self.T = dyn_sys.T

    def compute_estimate(self):
        self.estimate = compute_estimate(self.G,self.Htilde,self.T_obs,self.T,self.X0,self.P0,
            self.R,self.dyn_sys)

    def plot_estimate(self):
        plot_estimate(self)

    def plot_residuals(self):
        plot_residuals(self)

def derive_range_rate_s(range_s,g_cords,ns):
    '''
    Returns the symbolic range rate
    Parameters:
    ----------
    range_s : (symbolic matrix) range measurement
    g_cords : (list of strings) generalized coordinates
    ns : (dictionnary) already-defined symbols
    '''
    gen_cords = sym.Matrix(np.zeros(len(g_cords)))

    gen_vels = sym.Matrix(np.zeros(len(g_cords)))
    range_rate_s = sym.diff(range_s[0],ns['t'])
    for i in range(len(gen_cords)):
        gen_cords[i] = sym.symbols(g_cords[i], real = True)(ns['t'])
        gen_vels[i] = sym.symbols(g_cords[i]+'_dot', real = True)(ns['t'])
        range_rate_s = range_rate_s.subs(sym.diff(gen_cords[i],ns['t']),gen_vels[i])

    return range_rate_s 


def compute_estimate(G,Htilde,T_obs,T,X_hat0,P0,Rcov,dyn_sys):
    '''
    Compute the state deviation estimate, reference trajectory estimate
    and state estimate error covariance
    Parameters:
    -----------
    G : (function handle) state observation equations
    Htilde: (function handle) state observation matrix 
    T_obs: (l-by-1 np.array) observation times
    T: (N-by-1 np.array) time histories used in the true dynamics propagation
    X_hat0 : (n-by-1 np.array) initial reference trajectory estimate
    P0 : (n-by-n np.array) initial error covariance
    Rcov : (p-by-p np.array) observation error covariance matrix
    dyn_sys : (instance of DynamicalSystem) Dynamical System whose state components are
    estimated
    Returns:
    --------
    dict{'x_hat','X_hat','P_hat'} (dictionnary) estimated quantities

    '''
    N_obs = Rcov.shape[0]

    x_hat = np.zeros([len(X_hat0),len(T)])
    X_hat = np.zeros([len(X_hat0),len(T)])
    P_hat = np.zeros([len(X_hat0),len(X_hat0),len(T)])
    Y = np.zeros([N_obs,len(T)])
    y = np.zeros([N_obs,len(T)])

    # Initialization. Best a-priori is used for the initial guess
    X_hat[:,0] = X_hat0
    x_hat[:,0] = np.zeros(X_hat0.shape)
    P_hat[:,:,0] = P0

    # Main filter loop
    for i in range(len(dyn_sys.T)-1):
        X_Phi_0 = np.concatenate([X_hat[:,i],np.eye(len(X_hat0)).reshape([len(X_hat0)**2,])])

        # Time update
        [x_bar,X_bar,P_bar] = time_update(X_hat[:,i],x_hat[:,i],P_hat[:,:,i],T[i],
            T[i + 1 ],dyn_sys)

        # Measurement update
        [x_hat_i,X_hat_i,P_hat_i,Y_i,y_i] = measurement_update(T[i + 1],x_bar,X_bar,
            P_bar,Htilde,Rcov,G,dyn_sys.X.T[:,i + 1])


        x_hat[:,i + 1 ] = x_hat_i
        X_hat[:,i + 1] = X_hat_i
        P_hat[:,:,i + 1] = P_hat_i
        Y[:,i + 1] = Y_i.reshape(Y[:,i + 1].shape)
        y[:,i + 1] = y_i.reshape(y[:,i + 1].shape)

    return {'x_hat': x_hat,'X_hat': X_hat, 'P_hat': P_hat, 'Y':Y, 'y':y}



def time_update(X_hat_old,x_hat_old,P_old,t_old,t_next,dyn_sys):
    '''
    Extrapolates the state deviation estimate, the non-linear state estimate
    and the state error covariance matrix
    Parameters:
    -----------
    X_hat_old : (n-by-1 np.array) non-linear state estimate at the previous time
    x_hat_old : (n-by-1 np.array) state deviation estimate at the previous time
    P_old : (n-by-n np.array) state estimate covariance matrix at the previous time
    t_old : previous time
    t_next : next time
    dyn_sys : (Instance of DynamicalSystem) dynamical system whose state-space
        are being estimated
    Returns:
    -----------    
    x_bar :  (n-by-1 np.array) a-priori state deviation estimate 
    X_bar :  (n-by-1 np.array) a-priori non-linear state
    P_bar : (n-by-n np.array) a-priori state covariance matrix

    '''
    n = len(x_hat_old)
    X_Phi_0 = np.concatenate([X_hat_old,
        np.eye(len(X_hat_old)).reshape([np.eye(len(x_hat_old)).size,])])
    u = lambda x: 0*x

    X_Phi = odeint(dxdt_interface, X_Phi_0, [t_old,t_next], 
        args = (dyn_sys.Xdot,dyn_sys.F), rtol = 3e-14 , atol = 1e-16)

    Phi = X_Phi.T[ n : n * (n + 1) , - 1 ].reshape(np.eye(n).shape)
    x_bar = np.dot(Phi,x_hat_old)
    X_bar = X_Phi.T[0 : n, -1 ]
    
    P_bar = np.dot(Phi,np.dot(P_old,Phi.T)) + 0.000001 * np.eye(Phi.shape[0])
    return [x_bar,X_bar,P_bar]

def measurement_update(t,x_bar,X_bar,P_bar,Htilde,Rcov,G,X_true):
    '''
    Performs the measurement update in the Extended Kalman Filter formulation
    Parameters:
    ----------
    t : time update
    x_bar : (n-by-1 np.array) a-priori state deviation
    X_bar : (n-by-1 np.array) a-priori non-linear state 
    P_bar : (n-by-n np.array) a-priori state error covariance matrix
    Htilde : (function handle) state-observation matrix
    Rcov : (p-by-p) observation error covariance
    G : (function handle) state-observation equations
    X_true : (n-by-1 np.array) true non-linear state
    Returns:
    --------
    x_hat : (n-by-1 np.array)  state estimate
    X_hat : (n-by-1 np.array) non-linear state 
    P_bar : (n-by-n np.array) state error covariance matrix
    Y : (p-by-1 np.array) observation
    y : (p-by-1 np.array) pre-fit residuals

    '''

    # State-observation matrix
    Hm = Htilde(*time_state(t,X_bar))

    # Observation noise
    epsilon = np.random.multivariate_normal(np.zeros(Rcov.shape[0]), Rcov)

    # Observation
    Y = G(*time_state(t,X_true)) + np.array([epsilon]).T

    # Prefit residuals
    y =  Y - G(*time_state(t,X_bar))
   
    # Kalman Gain
    K1 = np.dot(P_bar,Hm.T)
    K2 = np.linalg.inv(np.dot(Hm,np.dot(P_bar,Hm.T)) + Rcov)
    K = np.dot(K1,K2)

   
    # Estimates update
    x_hat = x_bar + np.dot(K, y - 
        np.dot(Hm,x_bar.reshape(Hm.shape[1],1))).reshape(x_bar.shape)
    X_hat = X_bar 
    P_hat = np.dot(np.eye(len(X_hat)) - np.dot(K,Hm),P_bar)
    # P_hat = np.dot(P_hat,(np.eye(len(X_hat)) - np.dot(K,Hm)).T) + np.dot(K,np.dot(Rcov,K.T))

    return [x_hat,X_hat,P_hat,Y,y]

def state_obs_funs(g_cords,state_obs_eq):
    '''
    Returns function handles to the state-observation matrix and equations
    Parameters:
    -----------
    g_cords : (list of strings) generalized coordinates
    state_obs_eq : (symbolic matrix) symbolic observation equations
    Returns:
    -----------
    Htilde (function handle) : state observation matrix
    G : (function handle) : state observation equations
    '''
    state = sym.Matrix(np.zeros(2*len(g_cords)))
    time_state = sym.Matrix(np.zeros(1+2*len(g_cords)))
    t = sym.symbols('t', real = True)
    time_state[0] = t
    
    for i in range(len(g_cords)):
        state[i] = sym.symbols(g_cords[i], real = True)(t)
        state[i+len(g_cords)] = sym.symbols(g_cords[i] + '_dot', real = True)(t)
    for i in range(len(state)):
        time_state[i+1] = state[i]


    Htilde_s = state_obs_eq.jacobian(state)
  
    
    Htilde = sym.lambdify(time_state,Htilde_s, modules = 'numpy')
    G = sym.lambdify(time_state,state_obs_eq, modules = 'numpy')
    return [Htilde,G]
    

def dxdt_interface(X,t , dxdt, Amat, u = lambda x: 0*x):
    """
    Provides an interface between odeint and dxdt
    Parameters :
    ------------
    X : (n-by-1 np array) state
    t : time
    dxdt : (function handle) time derivative of the true (n-by-1) state vector
    A_mat : (function handle) state-space matrix
    u : (function handle) state-feedback control
    Returns:
    --------
    (n*(n+1)-by-1 np.array) time derivative of the components of the augmented state 
    """
    # Number of state parameters
    n = 0.5 * (-1 + np.sqrt(1+4*len(X)))
    Xdot = np.zeros([n*(n+1)])

    # Arguments
    args = np.concatenate([[t],X[0:n]])

    # State derivative 
    Xdot_state = dxdt(*list(args))
  

    # State-Space matrix
    A = Amat(*list(args))
    Xdot[0 : n] = Xdot_state.reshape([len(Xdot_state),])
    Xdot[n : n * (n+1) ] = A.reshape([A.size,])
    return Xdot


def plot_residuals(CKF):
    '''
    Plots the prefit and post-fit residuals
    Parameters:
    -----------
    CKF: (instance of ExtendedKalman)
    '''
    plt.clf()
    Rcov = CKF.R
    g_cords = CKF.dyn_sys.g_cords
    X = CKF.dyn_sys.X
    T = CKF.dyn_sys.T
    Y = CKF.estimate['Y']
    y = CKF.estimate['y']
    x_hat = CKF.estimate['x_hat']
    X_hat = CKF.estimate['X_hat']
    P_hat = CKF.estimate['P_hat']

    post_fit_res = np.zeros([CKF.N_obs,len(T)])
 
    for i in range(len(T)):
        post_fit_res[:,i] = y[:,i] - np.dot(CKF.Htilde(*time_state(T[i],x_hat[:,i])),x_hat[:,i])

    for i in range(CKF.N_obs):
        plt.plot(T, y[i,:]/np.sqrt(Rcov[i,i]),'.')
       
        plt.ticklabel_format(useOffset=False)
        plt.legend()

    plt.xlabel('Time (s)')
    plt.suptitle('Prefit residuals ratios')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5,forward=True)
    plt.savefig('prefit_residuals_ckf.pdf', bbox_inches='tight')


    plt.clf()
    for i in range(CKF.N_obs):
      
        plt.plot(T, post_fit_res[i,:]/np.sqrt(Rcov[i,i]),'.')
       
       
        plt.legend()

    plt.xlabel('Time (s)')
    plt.suptitle('Post-fit residuals ratios')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5,forward=True)
    plt.savefig('post_fit_residuals_ckf.pdf', bbox_inches='tight')

    plt.clf()
    for i in range(len(g_cords)):
        plt.subplot(2*len(g_cords), 1, i+1)

        plt.plot(T, np.sqrt(P_hat[i,i]),'b--')
        plt.plot(T, -np.sqrt(P_hat[i,i]),'b--')
        plt.plot(T, CKF.dyn_sys.X.T[i,:]-(X_hat[i,:] + x_hat[i,:]),color = 'b')


        plt.title('$' + g_cords[i]+ '$')
        plt.ylabel('$' + g_cords[i]+ '$')
        plt.ticklabel_format(useOffset=False)
        
        plt.subplot(2*len(g_cords), 1, i+len(g_cords)+1)
        plt.plot(T, np.sqrt(P_hat[i+len(g_cords),i+len(g_cords)]),'b--')
        plt.plot(T, -np.sqrt(P_hat[i+len(g_cords),i+len(g_cords)]),'b--')
        plt.plot(T, CKF.dyn_sys.X.T[i + len(g_cords),:] - 
            (X_hat[i + len(g_cords),:] + x_hat[i + len(g_cords),:]),color = 'b')
        

        plt.ticklabel_format(useOffset=False)
        plt.title('$\dot{'+ g_cords[i] + "}$")
        plt.ylabel('$\dot{'+ g_cords[i] + "}$")
        plt.legend()

    plt.xlabel('Time (s)')
    plt.suptitle('Covariance envellope')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5,forward=True)
    plt.savefig('state_estimate_covar_ckf.pdf', bbox_inches='tight')



def plot_estimate(CKF):
    """
    Plots the estimated and true states' history
    Parameters:
    -----------
    CKF: (instance of ClassicalKalman) 
    """
    plt.clf()

    g_cords = CKF.dyn_sys.g_cords
    X = CKF.dyn_sys.X
    T = CKF.dyn_sys.T
    X_hat = CKF.estimate['X_hat']
    x_hat = CKF.estimate['x_hat']


    for i in range(len(g_cords)):
        plt.subplot(2*len(g_cords), 1, i+1)

        plt.plot(T, X.T[i,:],label = "True")
        plt.plot(T, X_hat[i,:] + x_hat[i,: ],label = "Estimated")

        plt.ticklabel_format(useOffset=False)
        plt.title('$' + g_cords[i] + '$')
        plt.ylabel('$' + g_cords[i] + '$')
        plt.legend()
        plt.subplot(2 * len(g_cords), 1, i + len(g_cords) + 1)

        plt.plot(T, X.T[i + len(g_cords),:],label = "True")
        plt.plot(T, X_hat[i + len(g_cords),:] + x_hat[i + len(g_cords),: ],label = "Estimated")

        plt.ticklabel_format(useOffset=False)
        plt.title('$\dot{'+ g_cords[i] + "}$")
        plt.ylabel('$\dot{'+ g_cords[i] + "}$")
        plt.legend()

    plt.xlabel('Time (s)')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5,forward=True)
    plt.savefig('true_vs_estimated_states_ckf.pdf', bbox_inches='tight')
