import sys
import os
import numpy as np
import open3d as o3d
import time
import math
import operator
from matplotlib import pyplot as plt
import matplotlib.style as mplstyle

import loadData
import sortCar_yimju as socar
from TrackingModule_fusion import track
from curve import curve, invadeROI
from tensorflow.keras.models import load_model

import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#tf.compat.v1.enable_eager_execution()


import pandas as pd
import cv2

################################################################################
######################### Calibration Matrix ###################################

RT = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
               [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
               [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
               [0.0, 0.0, 0.0, 1.0]])

R_rect_00 = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0],
                      [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0],
                      [7.402527e-03, 4.351614e-03, 9.999631e-01, 0.0],
                      [0.0, 0.0, 0.0, 1]])

P_rect_00 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                      [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                      [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

Calibration_matrix = P_rect_00 @ R_rect_00 @ RT

###############################################################################


def save_to_csv(index, start, duration, state, framenum, path_csv):
    datalist = np.full((start,5),np.nan)
    datalist = np.append(datalist, state, axis = 0)
    datalist = np.append(datalist, np.full((framenum-start-duration+1, 5), np.nan), axis = 0)
    pd.DataFrame(datalist).to_csv(path_csv + '{}.csv'.format(index))


# scailing
def minmaxScailing(datalist, xmean, xstd, ymean, ystd, vmean, vstd):
    mean_array = np.array([xmean, ymean, vmean, 0, 0])
    std_array = np.array([xstd, ystd, vstd, 1, 1])
    for i in range(len(datalist)):
        datalist[i] = (datalist[i]-mean_array)/std_array
    return datalist

# Set Track list
Track_list = []
Track_list_valid = []
frame_num = 0

mplstyle.use('fast')
plt.ion()
plt.figure(figsize=(10, 70))
cv2.namedWindow('Show Image')

path = "/media/yimju/Samsung_T5/kitti_for_train/faraway_lanechange/"
path_lidar = path + "lidar/"
path_csv = path + "csvdata/"
path_image = path + "image/"
path_semanticMap = path + "binfile/"

model = load_model('./car_kitti_9steps.h5')
# model = load_model('/home/jinyoung/model/car_kitti_moreandmore_overfit.h5')

file_list = loadData.load_data(path_lidar)
image_list = loadData.load_data(path_image)
semantic_list = loadData.load_data(path_semanticMap)

dt = 0.1

for files in file_list:
    print("frame_num: ", frame_num)
    # if frame_num % 2 == 1:
    #     frame_num += 1
    #     continue

    starttime = time.time()
    car_centroid = np.empty([0,3])
    car_box = np.empty([0,3])
    car_id = []
    car_processed = []
    ped_centroid = np.empty([0,3])
    ped_box = np.empty([0,3])
    ped_id = []
    ped_processed = []
    else_centroid = np.empty([0,3])
    else_box = np.empty([0,3])
    else_id = []
    else_processed = []
    
    print('before cv2 imread')
    # Load Image , Semantic Map, Pointcloud
    img = cv2.imread(path_image + image_list[frame_num], cv2.IMREAD_COLOR)
    semanticMap = np.fromfile(path_semanticMap + semantic_list[frame_num], dtype = np.int64)
    semanticMap = semanticMap.reshape(img.shape[0], img.shape[1])
    data = np.fromfile(path_lidar+files,dtype = np.float32)
    data = data.reshape(-1,4)
    data = data[:, :3]

    


    ########################################################################
    ##################### Crop Pointcloud ##################################

    #  (-10 <= x <= 40 & -10 <= y <= 10)
    # Now, N by 4 matrix for future fusion
    data = data[(data[:,1] <= 20)]
    data = data[(data[:,1] >= -20)]
    data = data[(data[:,0] >= -10)]
    data = data[(data[:,0] <= 60)]

    ##############################
    # Downsampling
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    cloud_downsample = cloud.voxel_down_sample(voxel_size=0.05)
    data = np.asarray(cloud_downsample.points)

    # Plot all Pointcloud
    plt.plot(data[:,0], data[:,1], 'ko', markersize = 0.3)

    ##############################
    # Extract partial pointcloud for low level fusion
    # Make another coordinates : data[3] = 1
    data = np.insert(data, 3, 1, axis = 1)

    data_front = data[(data[:,0] >= 0)]
    data_back = data[(data[:,0] < 0)]

    # Pointcloud for lane segmentation
    pc_front_below = data_front[(data_front[:,2] < -1.3)]
    pc_below = pc_front_below[:,:3]
    # Pointcloud for obstacles
    pc_front = data_front[(data_front[:,2] >= -1.1)]
    pc_back = data_back[(data_back[:,2] >= -1.1)]


    pc_all = np.append(pc_back, pc_front, axis = 0)
    pc_all = pc_all[:,:3]
    #######################################################################



    ########################################################################
    ################ Low level Fusion : Get label ##########################
    

    ####### Calibration

    #################### 
    # # For lane detection
    print('before lane detection')
    Y_below = (Calibration_matrix @ pc_front_below.T).T
    Y_below[:,0] = Y_below[:,0] / Y_below[:,2]
    Y_below[:,1] = Y_below[:,1] / Y_below[:,2]
    Y_below = Y_below[:, :2]
    Y_below = np.int_(Y_below)
    label_below = np.array(list(map(lambda x : semanticMap[x[1], x[0]] if 0 <= x[0] < img.shape[1] and 0 <= x[1] < img.shape[0] else 0, Y_below)))
    
    pc_lane = pc_below[np.where(label_below == 24)[0]]
    plt.plot(pc_lane[:,0], pc_lane[:,1], 'co', markersize = 0.3)
    left_lane, right_lane, left_fit ,right_fit = curve(pc_lane)

    plt.plot(left_lane[:,0], left_lane[:,1],'r')
    plt.plot(right_lane[:,0],right_lane[:,1],'r')
    print('after plt plot left, right lane')

    ####################
    # For Obstacles
    # For upper and front side
    Y_front = (Calibration_matrix @ pc_front.T).T
    Y_front[:,0] = Y_front[:,0] / Y_front[:,2]
    Y_front[:,1] = Y_front[:,1] / Y_front[:,2]
    Y_front = Y_front[:, :2]
    Y_front = np.int_(Y_front)
    label_front = np.array(list(map(lambda x : semanticMap[x[1], x[0]] if 0 <= x[0] < img.shape[1] and 0 <= x[1] < img.shape[0] else 0, Y_front)))
    
    # For upper and back side
    # No semantic map for back side
    label_back = np.zeros(len(pc_back))

    pc_label = np.append(label_back, label_front, axis = 0)




    ########################################################################
    ############################ Clustering ################################
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_all)


    clustertime = time.time()
    labels = np.asanyarray(cloud.cluster_dbscan(0.9,4))
    print("Clustering time: ", time.time() - clustertime)

    print(pc_all.shape)
    for i in range(np.max(labels)+1):
        DBSCAN_Result = cloud.select_by_index(np.where(labels == i)[0])
        clusterCloud = np.asarray(DBSCAN_Result.points)

        if len(clusterCloud) <= 15: 
            continue
        plt.plot(clusterCloud[:,0], clusterCloud[:,1], 'co', markersize = 0.5)
        # Classification
        # Car : 55, Lane : 24, Human : 19, Bicycle : 52, else : 0 and else
        # Front partition
        # Back Partition -> not known
        classification = np.bincount( np.int_(pc_label[np.where(labels == i)[0]]) ).argmax()

        ########################################################################
        ######################## Get Center and Box ############################
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])

        center = DBSCAN_Result.get_center()

        # Get 4 Box Points
        box = DBSCAN_Result.get_oriented_bounding_box()
        box_center = box.center
        #plt.plot(box_center_plot[0], box_center_plot[1],'go', markersize = 1.5)
        box_center = box_center[:2]

        box_points = box.get_box_points()
        box_points_numpy = np.asarray(box_points)
        #print(box_points_numpy)
        box_points_numpy_plot = (np.array([[0,-1,0], [1,0,0], [0,0,1]]) @ box_points_numpy.T).T 
        #plt.plot(box_points_numpy_plot[:,0], box_points_numpy_plot[:,1], 'go', markersize = 1.5)
        # for i in range(0,8):
        #     plt.text(box_points_numpy_plot[i,0], box_points_numpy_plot[i,1], '{}'.format(i))
        box = np.array([[(box_points_numpy[0,0]+box_points_numpy[3,0])/2, (box_points_numpy[0,1]+box_points_numpy[3,1])/2],
                        [(box_points_numpy[2,0]+box_points_numpy[5,0])/2, (box_points_numpy[2,1]+box_points_numpy[5,1])/2],
                        [(box_points_numpy[4,0]+box_points_numpy[7,0])/2, (box_points_numpy[4,1]+box_points_numpy[7,1])/2],
                        [(box_points_numpy[1,0]+box_points_numpy[6,0])/2, (box_points_numpy[1,1]+box_points_numpy[6,1])/2]])
        box_plot = (np.array([[0,-1], [1,0,]]) @ box.T).T
        #plt.plot(box_plot[:,0], box_plot[:,1], 'ro', markersize = 3)

        # Sort box
        box = socar.sortline_angle(box, box_center)
        
        # Get length of 4 line
        width = 100
        length = 0
        yaw = 0
        yaw_norm = 0
        # width : min / length : max

        for i in range(0,len(box)-1):
            tmp = math.sqrt((box[i,0] - box[i+1,0])**2 + (box[i,1] - box[i+1,1])**2)

            if width > tmp:
                width = tmp
                yaw_norm = math.atan( (box[i,1] - box[i+1,1]) / (box[i,0] - box[i+1,0]) )
            if length < tmp:
                length = tmp
                yaw = math.atan( (box[i,1] - box[i+1,1]) / (box[i,0] - box[i+1,0]) )

        templist_res = [box_center[0], box_center[1], yaw]
        templist_box = [width, length, z_max - z_min]


        ########################################################################
        ###################### Save Measurement by class #######################
        # Car : 55, bus : 54, truck : 61 / Lane : 24 / Human : 19, Bicycle : 52 / forest : 30, else : 0 and else, 
        if classification == 55 or classification == 54 or classification == 61:
            car_centroid = np.append(car_centroid, [templist_res], axis = 0)
            car_box = np.append(car_box, [templist_box], axis = 0)
            car_id.append(i)
            #plt.plot(clusterCloud[:,0], clusterCloud[:,1], 'ro', markersize = 0.5)

        elif classification == 19 or classification == 52:
            ped_centroid = np.append(ped_centroid, [templist_res], axis = 0)
            ped_box = np.append(ped_box, [templist_box], axis = 0)
            ped_id.append(i)
            #plt.plot(clusterCloud[:,0], clusterCloud[:,1], 'go', markersize = 0.5)

        elif classification != 30: # exclude forests
            # Unrecognized -> sortcar 
            #plt.plot(clusterCloud[:,0], clusterCloud[:,1], 'co', markersize = 0.5)
            if (width >= 1.3 or length >= 1.3) and width<=4 and length <= 8:
                if math.fabs(yaw - yaw_norm) >= 70 * math.pi/180:
                    else_centroid = np.append(else_centroid, [templist_res], axis = 0)
                    else_box = np.append(else_box, [templist_box], axis = 0)
                    else_id.append(i)
    
    ########################################################################
    ##################### Tracking #########################################

    car_processed = np.zeros(len(car_id))
    ped_processed = np.zeros(len(ped_id))
    else_processed = np.zeros(len(else_id))

    ########### Track Update ############
    if Track_list:
            for i in range(0,len(Track_list)):
                if Track_list[i].dead_flag == 1:
                    continue
                Track_list[i].unscented_kalman_filter(car_centroid, car_box, car_processed, ped_centroid, ped_box, ped_processed, else_centroid, else_box, else_processed, dt)

    ########### Create Track ###########
    # For Car
    for i in range(0, len(car_centroid)):
        if car_processed[i] == 1:
            continue

        Track = track(car_centroid[i], car_box[i], frame_num, 0)
        Track_list.append(Track)

    # For Pedestrian
    for i in range(0, len(ped_centroid)):
        if ped_processed[i] == 1:
            continue

        Track = track(ped_centroid[i], ped_box[i], frame_num, 1)
        Track_list.append(Track)

    # For else case
    for i in range(0, len(else_centroid)):
        if else_processed[i] == 1:
            continue

        Track = track(else_centroid[i], else_box[i], frame_num, 2)
        Track_list.append(Track)
    

    ########## Track Management ########
    if Track_list:
        try:
            for i in range(0, len(Track_list)):

                # Dismiss DeadTrack
                if Track_list[i].dead_flag == 1:
                    continue

                # Activate Track
                if Track_list[i].Activated == 0 and Track_list[i].Age >= 3:
                    Track_list[i].Activated = 1
                
                # deActivate Track
                if Track_list[i].DelCnt >= 7:
                    Track_list[i].dead_flag = 1
                
                '''# Delete Track
                if Track_list[i].DelCnt >= 20:
                    #del Track_list[i]
                    Track_list[i].dead_flag = 1'''
                
                # Initialize Tracks' processed check
                #Track_list[i].processed = 0

                # deActivate Else Case
                if Track_list[i].classification == 2 and Track_list[i].state[0] >= 15:
                    Track_list[i].dead_flag = 1
        
        except:
            print("Track was deleted")

    for i in range(0, len(Track_list)):
        length_his_state = len(Track_list[i].history_state)
        if((Track_list[i].classification==2 or Track_list[i].classification==0) and Track_list[i].Activated==1 and Track_list[i].dead_flag==0 and length_his_state>=15):
            
            temp_state = Track_list[i].history_state[length_his_state-15:]
            final_state = minmaxScailing(temp_state.tolist(), 15.720, 13.890, 2.301, 3.142, -3.054, 8.282)
            X_test = np.array(final_state)[:,:,np.newaxis,np.newaxis]
            XLIST = np.squeeze(model.predict(X_test[np.newaxis, :, :]))[:,0]
            YLIST = np.squeeze(model.predict(X_test[np.newaxis, :, :]))[:,1]
            #Track_list[i].motionPredict = 1
            flag =1
        else:
            #Track_list[i].motionPredict = 0
            flag = 0
            # x = graph.get_tensor_by_name('x_:0')
            # feed_dict ={x:final_state.reshape(-1, 12,5)} 
            # op_to_restore = graph.get_tensor_by_name('pred:0')
            # car_state = np.argmax(sess.run(op_to_restore,feed_dict),axis=1)

            # if(car_state ==[1]):
            #     Track_list[i].motionPredict=1
            # elif(car_state==[2]):
            #     Track_list[i].motionPredict=2
            # print('deep learning time : ', time.time()-start_dl_time)

        if Track_list[i].Activated == 1 and Track_list[i].processed == 1:
            # temp = cloud_downsample.select_by_index(np.where(labels == Track_list[i].ClusterID)[0])
            # temp = np.asarray(temp.points)
            Track_list[i].processed = 0

            # if len(temp) == 0:
            #     continue

            #temp = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ temp.T).T
            center = np.array([Track_list[i].state[0], Track_list[i].state[1]])
            
            w_box = Track_list[i].width_max
            l_box = Track_list[i].length_max
            yaw_box = Track_list[i].state[3]
            #yaw_box = Track_list[i].yaw_angle
            # if(yaw_box>=0):
            #     rec_box_1 = np.array([center[0] + math.cos(yaw_box) * l_box / 2 - math.sin(yaw_box) * w_box / 2
            #                         ,center[1] + math.sin(yaw_box) * l_box / 2 + math.cos(yaw_box) * w_box / 2])
            #     rec_box_2 = np.array([center[0] - math.cos(yaw_box) * l_box / 2 - math.sin(yaw_box) * w_box / 2
            #                         ,center[1] - math.sin(yaw_box) * l_box / 2 + math.cos(yaw_box) * w_box / 2])
            #     rec_box_3 = np.array([center[0] - math.cos(yaw_box) * l_box / 2 + math.sin(yaw_box) * w_box / 2
            #                         ,center[1] - math.sin(yaw_box) * l_box / 2 - math.cos(yaw_box) * w_box / 2])
            #     rec_box_4 = np.array([center[0] + math.cos(yaw_box) * l_box / 2 + math.sin(yaw_box) * w_box / 2
            #                         ,center[1] + math.sin(yaw_box) * l_box / 2 - math.cos(yaw_box) * w_box / 2])

            # elif(yaw_box<0):
            #     rec_box_1 = np.array([center[0] - math.cos(yaw_box) * l_box / 2 - math.sin(yaw_box) * w_box / 2
            #                         ,center[1] - math.sin(yaw_box) * l_box / 2 + math.cos(yaw_box) * w_box / 2])
            #     rec_box_2 = np.array([center[0] - math.cos(yaw_box) * l_box / 2 + math.sin(yaw_box) * w_box / 2
            #                         ,center[1] - math.sin(yaw_box) * l_box / 2 - math.cos(yaw_box) * w_box / 2])
            #     rec_box_3 = np.array([center[0] + math.cos(yaw_box) * l_box / 2 + math.sin(yaw_box) * w_box / 2
            #                         ,center[1] + math.sin(yaw_box) * l_box / 2 - math.cos(yaw_box) * w_box / 2])
            #     rec_box_4 = np.array([center[0] + math.cos(yaw_box) * l_box / 2 - math.sin(yaw_box) * w_box / 2
            #                         ,center[1] + math.sin(yaw_box) * l_box / 2 + math.cos(yaw_box) * w_box / 2])
            
            # rec_box_1 = (np.array([[0,-1], [1,0]]) @ rec_box_1.T).T
            # rec_box_2 = (np.array([[0,-1], [1,0]]) @ rec_box_2.T).T
            # rec_box_3 = (np.array([[0,-1], [1,0]]) @ rec_box_3.T).T
            # rec_box_4 = (np.array([[0,-1], [1,0]]) @ rec_box_4.T).T

                        
            #center = (np.array([[0,-1], [1,0]]) @ center.T).T
            # if(abs(Track_list[i].state[2])>0.5):
            #     if(Track_list[i].motionPredict == 0):
            #         #plt.plot((rec_box_1[0], rec_box_2[0], rec_box_3[0], rec_box_4[0], rec_box_1[0]), (rec_box_1[1], rec_box_2[1], rec_box_3[1], rec_box_4[1], rec_box_1[1]), 'g')
            #         plt.plot(center[0], center[1], 'go', markersize=20)
            #     elif(Track_list[i].motionPredict==1):
            #         #plt.plot((rec_box_1[0], rec_box_2[0], rec_box_3[0], rec_box_4[0], rec_box_1[0]), (rec_box_1[1], rec_box_2[1], rec_box_3[1], rec_box_4[1], rec_box_1[1]), 'r')
            #         plt.plot(center[0], center[1], 'ro', markersize=20)
            #     elif(Track_list[i].motionPredict ==2):
            #         #plt.plot((rec_box_1[0], rec_box_2[0], rec_box_3[0], rec_box_4[0], rec_box_1[0]), (rec_box_1[1], rec_box_2[1], rec_box_3[1], rec_box_4[1], rec_box_1[1]), 'c')
            #         plt.plot(center[0], center[1], 'co', markersize=20)
            #     else:
            #         plt.plot((rec_box_1[0], rec_box_2[0], rec_box_3[0], rec_box_4[0], rec_box_1[0]), (rec_box_1[1], rec_box_2[1], rec_box_3[1], rec_box_4[1], rec_box_1[1]), 'Y')
            
            # plt.plot(temp[:,0], temp[:,1], 'ro', markersize = 0.4)
            # plt.plot(center[0], center[1], 'go')
            # Plot car

            #14.881, 13.291, 2.256, 3.230
            #15.720, 13.890, 2.301, 3.142
            if Track_list[i].classification == 0:
                plt.plot(center[0], center[1], 'ro', markersize = 10)
                if(flag==1):
                    XLIST = XLIST*13.890 + 15.720
                    YLIST = YLIST*3.142+2.301
                    plt.plot(XLIST, YLIST, 'ro', markersize = 5)
            elif Track_list[i].classification == 1:
                plt.plot(center[0], center[1], 'go', markersize = 10)
            elif Track_list[i].classification == 2:
                plt.plot(center[0], center[1], 'bo', markersize = 10)
                if(Track_list[i].motionPredict==1):
                    XLIST = XLIST*13.890 + 15.720
                    YLIST = YLIST*3.142+2.301
                    #plt.plot(XLIST, YLIST, 'b', linewidth=5)

            plt.text(center[0], center[1], 'Track{}'.format(i+1))

            #u, v = math.cos(Track_list[i].state[3]), math.sin(Track_list[i].state[3])

            if Track_list[i].state[2] >= 1:
                u, v = math.cos(Track_list[i].state[3]), math.sin(Track_list[i].state[3])
                #[u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
                #plt.quiver(center[0], center[1], u, v, scale= 3, scale_units = 'inches', color = 'red')
            elif Track_list[i].state[2] < -1:
                u, v = math.cos(Track_list[i].state[3] + math.pi), math.sin(Track_list[i].state[3] + math.pi)
                #[u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
                #plt.quiver(center[0], center[1], u, v, scale= 3, scale_units = 'inches', color = 'red')
            
            #[u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
            
            # Plot Track's trace
            #for j in range(0, len(Track_list[i].trace)):
            #    trace_for_plot
            #plt.plot(Track_list[i].trace_x[:], Track_list[i].trace_y[:], 'g')

        # Initialize Tracks' processed check
        
    #print("enumerate time: ", time.time() - starttime)

    # show image data



    plt.xlim(-10,50)
    plt.ylim(-30,30)
    plt.draw()
    plt.pause(0.001)
    plt.clf()

    print("One Iteration Time : ", time.time() - starttime)


    frame_num += 1


    # cv2.imshow("Show Image", img)
    # if cv2.waitKey(33) == ord('a'):
    #     continue
'''
for i in range(0, len(Track_list)):
        if Track_list[i].Activated == 1:
            Track_list_valid.append((Track_list[i], i+1))
    
validtracklistnum =len(Track_list_valid)

######################################
# Save csv Data

for i in range(validtracklistnum):
    index = Track_list_valid[i][1]
    start = Track_list_valid[i][0].Start
    duration = len(Track_list_valid[i][0].history_state)
    state = Track_list_valid[i][0].history_state
    framenum = frame_num
    save_to_csv(index, start, duration, state, framenum, path_csv)
    '''
