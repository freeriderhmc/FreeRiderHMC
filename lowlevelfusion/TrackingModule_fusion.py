########################### TrackingModule ###########################
import numpy as np
from numpy.linalg import inv, cholesky
import math as mt
from matplotlib import pyplot as plt

class track:
    # Initialization
    def __init__(self, centroid, box, frame_num, classification):
        self.state = centroid
        self.state = np.insert(self.state,2,0.1)
        self.state = np.insert(self.state,4,0.05)
        self.box = box
        self.P = np.array([[0.3, 0, 0, 0, 0],
                           [0, 0.3, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0.3, 0],
                           [0, 0, 0, 0, 0.3]])
        self.Q = np.array([[0.2, 0, 0, 0, 0],
                           [0, 0.2, 0, 0, 0],
                           [0, 0, 0.2, 0, 0],
                           [0, 0, 0, 0.01, 0],
                           [0, 0, 0, 0, 0.1]])
        self.R = np.array([[0.01, 0, 0, 0],
                           [0, 0.01, 0, 0],
                           [0, 0, 0.5, 0],
                           [0, 0, 0, 0.1]])
        self.width_max = 2
        self.length_max = 4
        self.processed = 0
        self.kappa = 0
        self.alpha = 0.3
        self.Age = 1
        self.classification = classification
        #self.ClusterID = cluster_id
        self.Start = frame_num
        self.Activated = 0
        self.DelCnt = 0
        self.history_state = np.empty([0,5])
        self.history_box = np.empty([0,3])
        self.dead_flag = 0
        self.motionPredict = 0
        
        ##
        self.yaw_angle = centroid[2]

    def sigma_points(self,P):
        # Should We take kappa len - 3 because of wid, len, hei term ??
        # wid, len, hei term -> not in the kalman filter but in the LPF ??
        n = len(self.state)
        Xi = np.zeros((n, 2*n+1))
        W = np.zeros(2*n+1)
        self.kappa=20
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
        # B = np.array([[1,0,0,0,0],
        #               [0,1,0,0,0],
        #               [0,0,1,0,0],
        #               [0,0,0,1,0],
        #               [0,0,0,0,1]])
        B = np.array([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0]])
        return B @ Xi

    #def unscented_kalman_filter(self, z_meas, box_meas, car_list, z_processed, dt):
    def unscented_kalman_filter(self, car_centroid, car_box, car_processed, ped_centroid, ped_box, ped_processed, else_centroid, else_box, else_processed, dt):
        temp = -1
        useWhatBox = -1
        """Unscented Ksalman Filter Algorithm."""
        # (1) Sample Sigma Points and Weights.
        Xi, W = self.sigma_points(self.P)
        # (2) Predict Mean and Error Covariance of States.
        fXi = self.fx(Xi, dt)
        x_pred, P_x = self.UT(fXi, W, self.Q)
        

        ######################################################################################
        # (3) Data Association
        ######################################################################################
        #  1) Car
        if self.classification == 0:
            # A) Use Car measurement
            # First Gate
            for i in range(0, len(car_centroid)):
                if car_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = car_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = car_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    car_processed[i] = 1
                    temp = i
                    useWhatBox = 0
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(car_centroid)):
                    if car_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = car_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = car_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        car_processed[i] = 1
                        temp = i
                        useWhatBox = 0
                        break
            
            # B) Use Else measurement
            # First Gate
            for i in range(0, len(else_centroid)):
                if else_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    else_processed[i] = 1
                    temp = i
                    useWhatBox = 2
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(else_centroid)):
                    if else_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        else_processed[i] = 1
                        temp = i
                        useWhatBox = 2
                        break

        ######################################################################################
        #  2) Pedestrian
        elif self.classification == 1:
            # A) Use Pedestrian measurement
            # First Gate
            for i in range(0, len(ped_centroid)):
                if ped_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = ped_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = ped_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    ped_processed[i] = 1
                    temp = i
                    useWhatBox = 1
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(ped_centroid)):
                    if ped_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = ped_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = ped_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        ped_processed[i] = 1
                        temp = i
                        useWhatBox = 1
                        break
            
            # B) Use Else measurement
            # First Gate
            for i in range(0, len(else_centroid)):
                if else_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    else_processed[i] = 1
                    temp = i
                    useWhatBox = 2
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(else_centroid)):
                    if else_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        else_processed[i] = 1
                        temp = i
                        useWhatBox = 2
                        break

        
        ######################################################################################
        #  3) Else : use car > pedestrian > else
        elif self.classification == 2:
            # A) Use Car measurement
            # First Gate
            for i in range(0, len(car_centroid)):
                if car_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = car_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = car_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    car_processed[i] = 1
                    temp = i
                    useWhatBox = 0
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(car_centroid)):
                    if car_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = car_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = car_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        car_processed[i] = 1
                        temp = i
                        useWhatBox = 0
                        break
            
            # B) Use Pedestrian measurement
            # First Gate
            for i in range(0, len(ped_centroid)):
                if ped_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = ped_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = ped_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    ped_processed[i] = 1
                    temp = i
                    useWhatBox = 1
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(ped_centroid)):
                    if ped_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = ped_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = ped_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        ped_processed[i] = 1
                        temp = i
                        useWhatBox = 1
                        break

            # C) Use Else measurement
            # First Gate
            for i in range(0, len(else_centroid)):
                if else_processed[i] == 1:
                    continue

                z_meas_trans = np.array([0,0])
                z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                        [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                z_meas_rot = Rot_inverse @ z_meas_trans
                
                if -self.width_max * 0.3 <= z_meas_rot[0] <= self.width_max * 0.3 and -self.length_max * 0.4 <= z_meas_rot[1] <= self.length_max * 0.4:
                    self.processed = 1
                    else_processed[i] = 1
                    temp = i
                    useWhatBox = 2
                    break

            # Second Gate (When Self Track is Not updated by the First gate)
            if self.processed == 0:
                for i in range(0, len(else_centroid)):
                    if else_processed[i] == 1:
                        continue

                    z_meas_trans = np.array([0,0])
                    z_meas_trans[0] = else_centroid[i,0] - x_pred[0]
                    z_meas_trans[1] = else_centroid[i,1] - x_pred[1]
                    # z_meas_trans[0] = clusters[i].res[0] - self.state[0]
                    # z_meas_trans[1] = clusters[i].res[1] - self.state[1]
                    Rot_inverse = np.array([[mt.cos(self.state[3]), mt.sin(self.state[3])],
                                            [-mt.sin(self.state[3]), mt.cos(self.state[3])]])
                    z_meas_rot = Rot_inverse @ z_meas_trans
                    
                    if -self.width_max * 0.5 <= z_meas_rot[0] <= self.width_max * 0.5 and -self.length_max * 0.9 <= z_meas_rot[1] <= self.length_max * 0.9:
                        self.processed = 1
                        else_processed[i] = 1
                        temp = i
                        useWhatBox = 2
                        break

            
        
        # (4) Measurement Update
        # No measurement
        if temp == -1:
            self.state = x_pred
            self.P = P_x
            self.DelCnt += 1
        
        # Measurement associated
        else:
            hXi = self.hx(fXi)
            z_pred, P_z = self.UT(hXi, W, self.R)

            # Calculate Off Diagonal Elements of Error Covariance and Kalman Gain.
            Pxz = W * (fXi - x_pred.reshape(-1, 1)) @ (hXi - z_pred.reshape(-1, 1)).T
            K = Pxz @ inv(P_z)

            # Validation Check : Yaw angle
            #heading_angle =
            if useWhatBox == 0:
                measured_state = car_centroid[temp]
            elif useWhatBox == 1:
                measured_state = ped_centroid[temp]
            elif useWhatBox == 2:
                measured_state = else_centroid[temp]
            # measured_state = centroid[temp]
            # self.yaw_angle = centroid[temp][2]

            # if self.length_max > box[temp][1]:
            #     if measured_state[0] > 5:
            #         measured_state[0] += (self.length_max - box[temp][1]) * mt.cos(self.state[3])
            #         measured_state[0] += (self.length_max - box[temp][1]) * mt.sin(self.state[3])
            #     elif measured_state[0] < -5:
            #         measured_state[0] -= (self.length_max - box[temp][1]) * mt.cos(self.state[3])
            #         measured_state[0] -= (self.length_max - box[temp][1]) * mt.sin(self.state[3])
                

            #print("measured angle:", measured_state[2])
            # if 80 * mt.pi/180 <= mt.fabs(measured_state[2] - self.state[3]) < 100*mt.pi/180:
            #     if self.state[3] >= 0:
            #         measured_state[2] += mt.pi/2
            #     elif self.state[3] < 0:
            #         measured_state[2] -= mt.pi/2
            
            # if mt.fabs(measured_state[2] - self.state[3]) >= 160*mt.pi/180:
            #     if self.state[3] >= 0:
            #         measured_state[2] -= mt.pi
            #     elif self.state[3] < 0:
            #         measured_state[2] += mt.pi
            
            # '''if measured_state[2] > mt.pi/2:
            #     measured_state[2] -= mt.pi
            # elif measured_state[2] < -mt.pi/2:
            #     measured_state[2] += mt.pi'''
            
            # if measured_state[2] > mt.pi/2 or measured_state[2] < -mt.pi/2:
            #     print("check")
            #     measured_state[2] = self.state[3]
            measured_state[2] = mt.atan((measured_state[1] - self.state[1]) / (measured_state[0] - self.state[0]))

            # if 80 * mt.pi/180 <= mt.fabs(measured_state[2] - self.state[3]) < 100*mt.pi/180:
            #     if self.state[3] >= 0:
            #         measured_state[2] += mt.pi/2
            #     elif self.state[3] < 0:
            #         measured_state[2] -= mt.pi/2

            measured_state = np.insert(measured_state, 2, (measured_state[0] - self.state[0])/mt.fabs(measured_state[0] - self.state[0])*mt.sqrt((self.state[0] - measured_state[0])**2 + (self.state[1] - measured_state[1])**2)/dt)
            #measured_state = np.insert(measured_state, 4, (measured_state[2] - self.state[3])/dt)


            self.state = x_pred + K @ (measured_state - z_pred)
            self.P = P_x - K @ P_z @ K.T

            if useWhatBox == 0:
                self.update_box(car_box[temp])
            elif useWhatBox == 1:
                self.update_box(ped_box[temp])
            elif useWhatBox == 2:
                self.update_box(else_box[temp])
            else:
                print("What's the matter??")

            self.Age += 1
            self.DelCnt = 0
            #self.ClusterID = cluster_id[temp]
            self.classification = useWhatBox
            

        
        
        # (5) Get max width and length box
        if self.width_max < self.box[0]:
            self.width_max = self.box[0]

        if self.length_max < self.box[1]:
            self.length_max = self.box[1]

        # (6) Store History
        self.history_state = np.append(self.history_state, [self.state], axis = 0)
        self.history_box = np.append(self.history_box, [self.box], axis = 0)

    def update_box(self, box_meas):
        self.box = (1 - self.alpha) * self.box + self.alpha * box_meas
