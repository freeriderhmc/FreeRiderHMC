import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky
import math as mt

def sigma_points(x_esti, Sigma, kappa):
    n = len(x_esti)
    Xi = np.zeros((n, 2*n+1))
    W = np.zeros(2*n+1)
    kappa=3-n
    Xi[:, 0] = x_esti
    W[0] = kappa / (n + kappa)
    
    U = cholesky((n + kappa)*Sigma)
    
    for i in range(n):
        Xi[:, i+1]   = x_esti + U[:, i]
        Xi[:, n+i+1] = x_esti - U[:, i]
        W[i+1]       = 1 / (2*(n+kappa))
        W[n+i+1]     = W[i+1]
        
    return Xi, W

def UT(Xi, W, noiseCov):
    mean = np.sum(W * Xi, axis=1)
    cov = W * (Xi - mean.reshape(-1, 1)) @ (Xi  - mean.reshape(-1, 1)).T
    return mean, cov + noiseCov

def fx(Xi):
    cosy=mt.cos(x_esti[2])
    siny=mt.sin(x_esti[2])
    A=np.array([[1,0,0,dt*cosy,0],
                [0,1,0,dt*siny,0],
                [0,0,1,0,dt],
                [0,0,0,1,0],
                [0,0,0,0,1]])
    return A @ Xi

def hx(Xi):
    B = np.array([[1,0,0,0,0],
                  [0,1,0,0,0],
                  [0,0,1,0,0]])
    return B @ Xi

def unscented_kalman_filter(z_meas, x_esti, P,WD,L):
    """Unscented Kalman Filter Algorithm."""
    # (1) Sample Sigma Points and Weights.
    Xi, W = sigma_points(x_esti, P, kappa)
    # (2) Predict Mean and Error Covariance of States.
    fXi = fx(Xi)
    x_pred, P_x = UT(fXi, W, Q)
    z_meas_trans=np.array([])
    z_meas_trans[0]=z_meas[0]-x_pred[0]
    z_meas_trans[1]=z_meas[1]-x_pred[1]
    Rot_inverse=np.array([[mt.cos(x_esti[2]),mt.sin(x_esti[2])],
                          [-mt.sin([x_esti[2]]),mt.cos(x_esti[2])]])
    z_meas_rot=Rot_inverse@z_meas_trans
    
    if -WD/2<z_meas_rot[0]<WD/2 and -L/2<z_meas_rot[1]<L/2:
      # (3) Calculate Mean and Error Covariance for the Expected Observation.
      hXi = hx(fXi)
      z_pred, P_z = UT(hXi, W, R)

      # (4) Calculate Off Diagonal Elements of Error Covariance and Kalman Gain.
      Pxz = W * (fXi - x_pred.reshape(-1, 1)) @ (hXi - z_pred.reshape(-1, 1)).T
      K = Pxz @ inv(P_z)

      # (5) Estimate Mean and Error Covariance of States. 
      x_esti = x_pred + K @ (z_meas - z_pred)
      P = P_x - K @ P_z @ K.T
      return x_esti, P
    else: return x_pred, P_x

# Input parameters.
time_end = 10
dt = 0.1

# Initialization for system model.
# Matrix: A, H, Q, R, P_0
# Vector: x_0
Q = np.array([[0.1, 0, 0, 0, 0],
              [0, 0.1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
R = np.array([[0.01, 0, 0],
              [0, 0.01, 0],
              [0, 0, 1]]) #

# Initialization for estimation.
x_0 = np.array([0, 0, 0, 0,0])  # [horizontal position, horizontal velocity, vertical position].
P_0 = 1 * np.eye(5)

# Initialization for sigma points.
kappa = 0

time = np.arange(0, time_end, dt)
n_samples = len(time)
xpos_esti_save = np.zeros(n_samples)
ypos_esti_save = np.zeros(n_samples)
vel_esti_save = np.zeros(n_samples)
yaw_esti_save = np.zeros(n_samples)
yawr_esti_save = np.zeros(n_samples)
xpos_meas_save = np.zeros(n_samples)
ypos_meas_save=np.zeros(n_samples)
yaw_meas_save=np.zeros(n_samples)

x_esti, P = None, None
x_t=np.array(range(100)) + np.random.normal(0,0.1**2)
y_t=x_t**2
vel_w = np.random.normal(0,0.1**2)
for i in range(0,n_samples):
    yaw_t=mt.atan2(2*x_t[i],1) + np.random.normal(0,0.09**2)
    z_meas=np.array([x_t[i],y_t[i],yaw_t])
    if i == 0:
        v_t=0
        x_esti, P = x_0, P_0
    else:
        v_t=mt.sqrt(1+(y_t[i]-y_t[i-1])**2)/dt
        x_esti, P = unscented_kalman_filter(z_meas, x_esti, P)
    print('x_esti: ',x_esti)
    print('z_meas: ',z_meas)

    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[1]
    yaw_esti_save[i] = x_esti[2]
    vel_esti_save[i] = x_esti[3]
    yawr_esti_save[i] = x_esti[4]
    xpos_meas_save[i] = z_meas[0]
    ypos_meas_save[i]=z_meas[1]
    yaw_meas_save[i]=z_meas[2]

