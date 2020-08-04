import sys
import os
import numpy as np
import open3d as o3d
import time
import math
import operator
from matplotlib import pyplot as plt

import loadData
import sortCar_yimju as socar
from TrackingModule_for_clusterclass import track
# from TrackingModule_for_clusterclass2 import track
from clusterClass import clusterClass

import pandas as pd

# Set Track list
Track_list = []
Track_list_valid = []
frame_num = 0


plt.figure()
plt.ion()
plt.show()


path = "/media/jinwj1996/Samsung_T5/Lyft_train/10/lidar/"

file_list = loadData.load_data(path)

for files in file_list:
    clusterClass_list = [] 
    dt = 0.2

    data = np.fromfile(path+files,dtype = np.float32)
    data = data.reshape(-1,5)
    data = data[:,0:3]
    data = (np.array([[math.cos(177*math.pi/180),-math.sin(177*math.pi/180),0], [math.sin(177*math.pi/180),math.cos(177*math.pi/180),0], [0,0,1]]) @ data.T).T

    data = data[(data[:,0] <= 60)]
    data = data[(data[:,0] >= -20)]
    data = data[(data[:,1] <= 20)]
    data = data[(data[:,1] >= -20)]

    data = data[((data[:, 2] >= -1.2))] # -1.56
    data = data[((data[:, 2] <= 1.5))] # -1.56

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)

    cloud_downsample = cloud.voxel_down_sample(voxel_size=0.05)

    cloud_downsample_plot = np.asarray(cloud_downsample.points)
    cloud_downsample_plot = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ cloud_downsample_plot.T).T    
    plt.plot(cloud_downsample_plot[:,0], cloud_downsample_plot[:,1],'ko', markersize = 0.4)

    labels = np.asanyarray(cloud_downsample.cluster_dbscan(0.7,3))

    for i in range(np.max(labels)+1):

        DBSCAN_Result = cloud_downsample.select_by_index(np.where(labels == i)[0])
        clusterCloud = np.asarray(DBSCAN_Result.points)
        
        if len(clusterCloud) <= 10: 
            continue
        
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])

        center = DBSCAN_Result.get_center()

        clusterCloud_plot = (np.array([[0,-1,0], [1,0,0], [0,0,1]]) @ clusterCloud.T).T 
        plt.plot(clusterCloud_plot[:,0], clusterCloud_plot[:,1],'bo', markersize = 0.4)

        # center_plot = (np.array([[0,-1,0], [1,0,0], [0,0,1]]) @ center.T).T
        # plt.plot(center_plot[0], center_plot[1],'ro', markersize = 0.8)

        # Get 4 Box Points
        box = DBSCAN_Result.get_oriented_bounding_box()
        box_center = box.center
        box_center_plot = (np.array([[0,-1,0], [1,0,0], [0,0,1]]) @ box_center.T).T
        #plt.plot(box_center_plot[0], box_center_plot[1],'go', markersize = 1.5)
        box_center = box_center[:2]

        box_points = box.get_box_points()
        box_points_numpy = np.asarray(box_points)
        print(box_points_numpy)
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
        
        if math.fabs(yaw - yaw_norm) >= math.pi/3:
            cluster = clusterClass(np.array(templist_res), np.array(templist_box), i, 1)
            clusterClass_list.append(cluster)
    
        else:
            cluster = clusterClass(np.array(templist_res), np.array(templist_box), i, 0)
            clusterClass_list.append(cluster)


    ########### Track Update ############
    if Track_list:
        for i in range(0,len(Track_list)):
            if Track_list[i].dead_flag == 1:
                continue
            Track_list[i].unscented_kalman_filter(clusterClass_list, dt)

    ########### Create Track ###########
    
    for i in range(0, len(clusterClass_list)):
        if clusterClass_list[i].processed == 1 or clusterClass_list[i].car_flag == 0:
            continue
        
        # z_meas[i] that are not used : Create new track
        clusterClass_list[i].processed = 1
        Track = track(clusterClass_list[i], frame_num, i)
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
        
        except:
            print("Track was deleted")

        

    
    # cloud_downsample_plot = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ cloud_downsample.T).T
    # # Plot all points
    # plt.xlim(-40,40)
    # plt.ylim(-20,60)
    # plt.plot(cloud_downsample_plot[:,0], cloud_downsample_plot[:,1],'ko', markersize = 0.4)
    # plt.text(-40, 20, '{}-th frame'.format(frame_num))

    for i in range(0, len(Track_list)):
        #print(Track_list[i].Activated, Track_list[i].processed)
        if Track_list[i].Activated == 1 and Track_list[i].processed == 1:
            temp = cloud_downsample.select_by_index(np.where(labels == Track_list[i].ClusterID)[0])
            temp = np.asarray(temp.points)

            if len(temp) == 0:
                continue

            plt.xlim(-40,40)
            plt.ylim(-20,60)
            temp = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ temp.T).T
            center = np.array([Track_list[i].state[0], Track_list[i].state[1]])
            center = (np.array([[0,-1], [1,0]]) @ center.T).T
            #plt.plot(temp[:,0], temp[:,1], 'ro', markersize = 0.4)
            # plt.plot(center[0], center[1], 'go')
            plt.text(center[0], center[1], 'Track{}'.format(i+1))

            u, v = math.cos(Track_list[i].state[3]), math.sin(Track_list[i].state[3])
            [u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
            plt.quiver(center[0], center[1], u, v, scale= 3, scale_units = 'inches', color = 'red')
            # Plot Track's trace
            #for j in range(0, len(Track_list[i].trace)):
            #    trace_for_plot
            #plt.plot(Track_list[i].trace_x[:], Track_list[i].trace_y[:], 'g')

        # Initialize Tracks' processed check
        Track_list[i].processed = 0

    plt.draw()
    plt.pause(0.001)
    plt.clf()
    # plot Ego Vehicle
    '''plt.text(0, 0, 'EgoCar')
    plt.plot(res[:,0], res[:,1], 'ro')
    for i in range(0, len(Track_list)):
        if Track_list[i].Activated == 1 and Track_list[i].dead_flag == 0:
            plt.plot(Track_list[i].state[0], Track_list[i].state[1], 'b*')
            plt.text(Track_list[i].state[0], Track_list[i].state[1], 'Track{}'.format(i+1))
            
            # Plot Track's trace
            #for j in range(0, len(Track_list[i].trace)):
            #    trace_for_plot
            #plt.plot(Track_list[i].trace_x[:], Track_list[i].trace_y[:], 'g')             
    plt.show()'''
    
    #for i in range(0, len(Track_list)):
    #    print("Track value: ".format(i), Track_list[i].state)

    #pre_time_stamp = time_stamp
    frame_num += 1    
    #input("Press Enter to continue...")

for i in range(0, len(Track_list)):
    if Track_list[i].Activated == 1:
        Track_list_valid.append(Track_list[i])

validtracklistnum =len(Track_list_valid)
print("# of all track_list : ", len(Track_list))
print("# of valid track_list : ", validtracklistnum)

    # '''for i in range(validtracklistnum):
    # index = i+1
    # start = Track_list_valid[i].Start
    # duration = len(Track_list_valid[i].history_state)
    # state = Track_list_valid[i].history_state
    # framenum = frame_num
    # save_to_csv(index, start, duration, state, framenum, path)'''
    
# reach to csv file
# csv_file = pd.read_csv('{}.csv'.format(i), index_col=0) # set i~


    # '''for i in range(len(Track_list)):
    # print("Track_list {}-th all state".format(i+1))
    # print(Track_list[i].history_state)
    # plt.figure()
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,0], label = 'x_point', color = 'b')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,1], label = 'y_point', color = 'g')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,2], label = 'velocity', color = 'r')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,3], label = 'yaw-angle', color = 'c')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,4], label = 'yaw_rate', color = 'm')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_box[:,0], label = 'width', color = 'y')
    # plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_box[:,1], label = 'length', color = 'k')
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # plt.show()'''

        

    