import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class ExtendedKalman:
    """Defines the ExtendedKalman Class"""
    def __init__(self, X0,P0,R,dyn_sys,state_obs):
        self.X0 = X0
        self.P0 = P0
        self.R = R
        self.dyn_sys = dyn_sys
        self.A = dyn_sys.F
        self.H = np.eye(len(X0))
        self.state_obs = state_obs
        self.T_obs = dyn_sys.T
        self.T = dyn_sys.T

    def state_obs_mat(self):
        self.H = state_obs_mat(self.dyn_sys.ns,
            self.dyn_sys.g_cords,self.state_obs)

    def compute_estimate(self):
        self.estimate = compute_estimate(self.T_obs,self.T,self.X0,self.P0,
            self.R,self.dyn_sys)

    def time_update(self):
        [self.x_bar,self.P_bar] = time_update(self.x_hat_old,
            self.P_old)
        
    def measurement_update(self):
        [self.x_hat,self.P] = measurement_update(self.x_bar,self.P_bar)
        self.P_old = self.P
        self.x_hat_old = self.x_hat

    def plot_estimate(self):
        plot_estimate(self.T,self.estimate)




def measurement_update(t,x_bar,P_bar,Htilde):
    H = Htilde(t,x,y,z,xdot,ydot,zdot,xs,ys,zs)
    K1 = np.dot(P_bar,H.T)
    K2 = np.linalg.inv(np.dot(H,np.dot(P_bar,H.T)) + Rcov)
    K = np.dot(K1,K2)

def time_update(X_hat_old,x_hat_old,P_old,t_old,t_next,dyn_sys):
    n = len(x_hat_old)
    X_Phi_0 = np.concatenate([X_hat_old,
        np.eye(len(X_hat_old)).reshape([np.eye(len(x_hat_old)).size,])])
    u = lambda x: 0*x

    X_Phi = odeint(dxdt_interface, X_Phi_0, [t_old,t_next], 
        args = (dyn_sys.Xdot,dyn_sys.F), rtol = 3e-14 , atol = 1e-16)

    Phi = X_Phi.T[ n : n * (n + 1) ,-1].reshape(np.eye(n).shape)
    x_bar = np.dot(Phi,x_hat_old)
    X_bar = X_Phi.T[0 : n, -1 ]
    P_bar = np.dot(Phi,np.dot(P_old,Phi.T))
    return [x_bar,X_bar,P_bar]


def compute_estimate(T_obs,T,X0,P0,R,dyn_sys):
    x_hat = np.zeros([len(X0),len(T)])
    X_hat = np.zeros([len(X0),len(T)])
    P_hat = np.zeros([len(X0),len(X0),len(T)])

    # Initialization. Best a-priori is used for the initial guess
    X_hat[:,0] = X0
    x_hat[:,0] = np.zeros(X0.shape)
    P_hat[:,:,0] = P0

    # Filter loop
    for i in range(len(dyn_sys.T)-1):
        X_Phi_0 = np.concatenate([X_hat[:,i],np.eye(len(X0)).reshape([len(X0)**2,])])
        [x_bar,X_bar,P_bar] = time_update(X_hat[:,i],x_hat[:,i],P_hat[:,:,i],T[i],
            T[i+1],dyn_sys)
        x_hat[:,i+1] = x_bar
        X_hat[:,i+1] = X_bar + x_bar
        P_hat[:,:,i+1] = P_bar

    return {'x_hat': x_hat,'X_hat': X_hat, 'P_hat': P_hat}

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
    args = np.concatenate([[t],X[0:n]])

    # State derivative 
    Xdot_state = dxdt(*list(args))
  

    # State-Space matrix
    A = Amat(*list(args))
    Xdot[0 : n] = Xdot_state.reshape([len(Xdot_state),])
    Xdot[n : n * (n+1) ] = A.reshape([A.size,])
    return Xdot