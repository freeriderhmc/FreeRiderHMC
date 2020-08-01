########################### TrackingModule ###########################
import numpy as np
from numpy.linalg import inv, cholesky
import math as mt
from clusterClass import clusterClass

class track:
    # Initialization
    def __init__(self, cluster, frame_num, id):
        self.state = cluster.res
        self.state = np.insert(self.state,2,1)
        self.state = np.insert(self.state,4,0.1)
        self.box = cluster.box
        self.P = np.array([[0.3, 0, 0, 0, 0],
                           [0, 0.3, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0.3, 0],
                           [0, 0, 0, 0, 1]])
        self.Q = np.array([[0.1, 0, 0, 0, 0],
                           [0, 0.1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0.1, 0],
                           [0, 0, 0, 0, 1]])
        self.R = np.array([[0.05, 0, 0],
                           [0, 0.05, 0],
                           [0, 0, 0.5]])
        self.width_max = 0
        self.length_max = 0
        self.processed = 0
        self.kappa = 0
        self.alpha = 0.3
        self.Age = 1
        self.ClusterID = id
        self.Start = frame_num
        self.Activated = 0
        self.DelCnt = 0
        #self.history_state = np.empty([0,5])
        #self.history_box = np.empty([0,3])
        self.dead_flag = 0        


    def sigma_points(self,P):
        # Should We take kappa len - 3 because of wid, len, hei term ??
        # wid, len, hei term -> not in the kalman filter but in the LPF ??
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
        '''
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
        '''
        # Revised CTRV Model
        ##################################
        # Use state of sigma points
        ##################################
        Xi_pred = np.zeros([5,11])

        for i in range(0,len(Xi.T)):
            coeff = Xi[2][i]/Xi[4][i]
            add_part = np.array([coeff * (mt.sin(Xi[3][i] + Xi[4][i]*dt) - mt.sin(Xi[3][i])),
                                 coeff * (-mt.cos(Xi[3][i] + Xi[4][i]*dt) + mt.cos(Xi[3][i])),
                                 0,
                                 Xi[4][i] * dt,
                                 0])
            Xi_pred[:,i] = Xi[:,i] + add_part



        '''coeff = self.state[2]/self.state[4]
        add_part = np.array([coeff * (mt.sin(self.state[3] + self.state[4]*dt) - mt.sin(self.state[3])),
                             coeff * (-mt.cos(self.state[3] + self.state[4]*dt) + mt.cos(self.state[3])),
                             0,
                             self.state[4] * dt,
                             0,
                             0,
                             0,
                             0])'''

        #return Xi + add_part.reshape(-1,1)
        return Xi_pred

    def hx(self,Xi):
        B = np.array([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,0,1,0]])
        return B @ Xi

    #def unscented_kalman_filter(self, z_meas, box_meas, car_list, z_processed, dt):
    def unscented_kalman_filter(self, clusters, dt):
        temp = -1
        """Unscented Kalman Filter Algorithm."""
        # (1) Sample Sigma Points and Weights.
        Xi, W = self.sigma_points(self.P)
        # (2) Predict Mean and Error Covariance of States.
        fXi = self.fx(Xi, dt)
        x_pred, P_x = self.UT(fXi, W, self.Q)
        
        # (3) Data Association
        ##### 1) Activated == 0 : Use only (car_flag == 0) cluster
        if self.Activated == 0:
            # First Gate
            for i in range(0, len(clusters)):
                if clusters[i].processed == 1 or clusters[i].car_flag == 0:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = clusters[i].res[0] - x_pred[0]
                z_meas_trans[1] = clusters[i].res[1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max/2 < z_meas_rot[0] < self.width_max/2 and -self.length_max/2 < z_meas_rot[1] < self.length_max/2:
                    self.processed = 1
                    clusters[i].processed = 1
                    temp = i
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(clusters)):
                    if clusters[i].processed == 1 or clusters[i].car_flag == 0:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = clusters[i].res[0] - x_pred[0]
                    z_meas_trans[1] = clusters[i].res[1] - x_pred[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 1 < z_meas_rot[0] < self.width_max * 1 and -self.length_max * 1 < z_meas_rot[1] < self.length_max * 1:
                        self.processed = 1
                        clusters[i].processed = 1
                        temp = i
                        break
        ##### 2) Activated == 1 : Use All Clusters
        else:
            # First Gate
            for i in range(0, len(clusters)):
                if clusters[i].processed == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = clusters[i].res[0] - x_pred[0]
                z_meas_trans[1] = clusters[i].res[1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max/2 < z_meas_rot[0] < self.width_max/2 and -self.length_max/2 < z_meas_rot[1] < self.length_max/2:
                    self.processed = 1
                    clusters[i].processed = 1
                    temp = i
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(clusters)):
                    if clusters[i].processed == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = clusters[i].res[0] - x_pred[0]
                    z_meas_trans[1] = clusters[i].res[1] - x_pred[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 1 < z_meas_rot[0] < self.width_max * 1 and -self.length_max * 1 < z_meas_rot[1] < self.length_max * 1:
                        self.processed = 1
                        clusters[i].processed = 1
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
            self.state = x_pred + K @ (clusters[temp].res - z_pred)
            self.P = P_x - K @ P_z @ K.T
            self.update_box(clusters[temp].box)
            self.Age += 1
            self.DelCnt = 0
            self.ClusterID = clusters[temp].id
        
        # Get max width and length box
        if self.width_max < self.box[0]:
            self.width_max = self.box[0]

        if self.length_max < self.box[1]:
            self.length_max = self.box[1]

        # Store History
        # self.history_state = np.append(self.history_state, [self.state], axis = 0)
        # self.history_box = np.append(self.history_box, [self.box], axis = 0)

    def update_box(self, box_meas):
        self.box = (1 - self.alpha) * self.box + self.alpha * box_meas

            