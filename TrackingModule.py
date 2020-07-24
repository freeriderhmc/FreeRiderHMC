########################### TrackingModule ###########################
import numpy as np
from numpy.linalg import inv, cholesky
import math as mt

class track:
    # Initialization
    def __init__(self, x_0):
        self.state = x_0
        self.state = np.insert(self.state,2,0)
        self.state = np.insert(self.state,4,0)
        self.P = np.eye(8)
        self.Q = np.array([[0.1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0.1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0.1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0.1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0.1]])
        self.R = np.array([[0.01, 0, 0, 0, 0, 0],
                          [0, 0.01, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0.1, 0, 0],
                          [0, 0, 0, 0, 0.1, 0],
                          [0, 0, 0, 0, 0, 0.1]])
        self.processed = 0
        self.kappa = 0
        self.Age = 1
        self.Activated = 0
        self.DelCnt = 0
        self.trace_x = []
        self.trace_y = []


    def sigma_points(self,P):
        n = len(self.state)
        Xi = np.zeros((n, 2*n+1))
        W = np.zeros(2*n+1)
        self.kappa=3
        Xi[:, 0] = self.state
        W[0] = self.kappa / (n + self.kappa)
        
        U = cholesky((n + self.kappa)*P)
      
        for i in range(n):
            Xi[:, i+1]   = self.state + U[:, i]
            Xi[:, n+i+1] = self.state - U[:, i]
            W[i+1]       = 1 / (2*(n+self.kappa))
            W[n+i+1]     = W[i+1]
            
        return Xi, W

    def UT(self,Xi, W, noiseCov):
        mean = np.sum(W * Xi, axis=1)
        cov = W * (Xi - mean.reshape(-1, 1)) @ (Xi  - mean.reshape(-1, 1)).T
        return mean, cov + noiseCov

    def fx(self,Xi, dt):
        cosy=mt.cos(self.state[2])
        siny=mt.sin(self.state[2])
        A=np.array([[1,0,dt*cosy,0,0,0,0,0],
                    [0,1,dt*siny,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,1,dt,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,1]])
        return A @ Xi

    def hx(self,Xi):
        B = np.array([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,1]])
        return B @ Xi

    def unscented_kalman_filter(self, z_meas, z_processed, dt):
        temp = -1
        """Unscented Kalman Filter Algorithm."""
        # (1) Sample Sigma Points and Weights.
        Xi, W = self.sigma_points(self.P)
        # (2) Predict Mean and Error Covariance of States.
        fXi = self.fx(Xi, dt)
        x_pred, P_x = self.UT(fXi, W, self.Q)
        
        # (3) Data Association
        # First Gate
        for i in range(0, len(z_meas)):
            if z_processed[i] == 1:
                continue

            z_meas_trans = np.array([0,0])
            z_meas_trans[0] = z_meas[i][0] - x_pred[0]
            z_meas_trans[1] = z_meas[i][1] - x_pred[1]
            Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                    [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
            z_meas_rot = Rot_inverse @ z_meas_trans
            
            if -self.state[5]/2 < z_meas_rot[0] < self.state[5]/2 and -self.state[6]/2 < z_meas_rot[1] < self.state[6]/2:
                self.processed = 1
                z_processed[i] = 1
                temp = i
                break

        # Second Gate (Not updated by the First gate)
        if self.processed == 0:
            for i in range(0, len(z_meas)):
                if z_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = z_meas[i][0] - x_pred[0]
                z_meas_trans[1] = z_meas[i][1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.state[5] * 1.5 < z_meas_rot[0] < self.state[5] * 1.5 and -self.state[6] * 1.5 < z_meas_rot[1] < self.state[6] * 1.5:
                    self.processed = 1
                    z_processed[i] = 1
                    temp = i
                    break


        if temp == -1:
            self.state = x_pred
            self.P = P_x
            self.DelCnt += 1
        else:
            hXi = self.hx(fXi)
            z_pred, P_z = self.UT(hXi, W, self.R)

            # (5) Calculate Off Diagonal Elements of Error Covariance and Kalman Gain.
            Pxz = W * (fXi - x_pred.reshape(-1, 1)) @ (hXi - z_pred.reshape(-1, 1)).T
            K = Pxz @ inv(P_z)
            self.state = x_pred + K @ (z_meas[temp] - z_pred)
            self.P = P_x - K @ P_z @ K.T
            self.Age += 1
            self.DelCnt = 0

            