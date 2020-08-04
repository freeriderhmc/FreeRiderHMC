#############################################################################
############################ For Panda Dataset ##############################
#############################################################################
import pandaset
import sys
import os
import numpy as np
import open3d as o3d
import time
import math
from math import cos, sin, pi
import operator
from matplotlib import pyplot as plt

import loadData
# import sortCar_for_clusterClass as socar
import sortCar_getresult as socar
from TrackingModule_for_clusterclass import track
# from TrackingModule_for_clusterclass2 import track
from clusterClass import clusterClass

import pandas as pd

###########################################################
###################### Setting ############################
###########################################################

pi = 3.141592653589793238

def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    return angle

def save_to_csv(index, start, duration, state, framenum, path):
    datalist = np.full((start,5),np.nan)
    datalist = np.append(datalist, state, axis = 0)
    datalist = np.append(datalist, np.full((framenum-start-duration+1, 5), np.nan), axis = 0)
    pd.DataFrame(datalist).to_csv(path_csv + '{}.csv'.format(index))


# Set mod
mod = sys.modules[__name__]

# Set Track list
Track_list = []
Track_list_valid = []

# Expand iteration limit
sys.setrecursionlimit(5000)

# Set Car Standard
carz_min, carz_max = 0, 3
carx_min, carx_max = 0, 7
cary_min, cary_max = 0, 7

# Set Visualizer and Draw x, y Axis
# vis = o3d.visualization.Visualizer()
# vis.create_window()

Axis_Points = [[0,0,0], [20,0,0],[0,20,0]]
Axis_Lines = [[0,1],[0,2]]

colors = [[0,0,0] for i in range(len(Axis_Lines))]

line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(Axis_Points), lines = o3d.utility.Vector2iVector(Axis_Lines))
line_set.colors = o3d.utility.Vector3dVector(colors)


line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(Axis_Points), lines = o3d.utility.Vector2iVector(Axis_Lines))
line_set.colors = o3d.utility.Vector3dVector(colors)


# Load binary data
# path = "/media/yimju/E/FreeRider/pandaset_0"
path = "/media/yimju/7B22CE2F737C2A58/Lyft_train/26/lidar/"
path_csv = "/media/yimju/7B22CE2F737C2A58/Lyft_train/26/csvdata/"
# f = open("./2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/timestamps.txt","r")


file_list = loadData.load_data(path)
frame_num = 0
pre_time_stamp = 0


# Prepare Pyplot Visualizer
# Visualization
plt.figure()
plt.ion()
plt.show()


############################################################################
################################ Main Loop #################################
############################################################################

# folder = '020'
# dataset = pandaset.DataSet("/media/yimju/E/FreeRider/pandaset_0")
# f = open(path+"/{}/lidar/timestamps.json".format(folder),"r")
# line = f.read()
# line = line[1:len(line)-1]
# line = line.split(', ')
#print(line)
# for folder in range(1,50):
        
# seq002 = dataset[folder]
# seq002.load_lidar().load_semseg()

# for seq_idx in range(0,80):  
for files in file_list:  
    clusterClass_list = []  
    # time_stamp = float(line[seq_idx])
    # dt = time_stamp-pre_time_stamp
    dt = 0.2

    # print("{}th Frame".format(frame_num + 1))
    # # get Pandar64 points
    # seq002.lidar.set_sensor(0)
    # pandar64_points = seq002.lidar[seq_idx].to_numpy()
    # data = pandar64_points[:, :3]
    
    data = np.fromfile(path+files,dtype = np.float32)
    data = data.reshape(-1,5)
    data = data[:,0:3]
    
    # Rotate 180 degree
    # Rotate 185 degree
    #data = (np.array([[-1, 0, 0], [0,-1,0], [0,0,1]]) @ data.T).T
    data = (np.array([[math.cos(177*math.pi/180),-math.sin(177*math.pi/180),0], [math.sin(177*math.pi/180),math.cos(177*math.pi/180),0], [0,0,1]]) @ data.T).T

   # Convert pcd to numpy array
    cloud_downsample = data

    # Crop Pointcloud -20m < x < 20m && -20m < y < 20m && z > -1.80m
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] <= 60))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] >= -30))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] <= 20))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] >= -20))]


    # Convert numpy into pointcloud 
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_downsample)
    

    # Downsampling pointcloud
    cloud_downsample = cloud.voxel_down_sample(voxel_size=0.05)
    cloud_downsample = np.asarray(cloud_downsample.points)


    # Plot all points
    cloud_downsample_plot = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ cloud_downsample.T).T    
    plt.xlim(-40,40)
    plt.ylim(-20,60)
    plt.plot(cloud_downsample_plot[:,0], cloud_downsample_plot[:,1],'ko', markersize = 0.4)
    plt.text(-40, 20, '{}-th frame'.format(frame_num))


    # threshold z value cut the road
    # KITTI : -1.3
    # Nuscenes : -1.0
    cloudoutliers = cloud_downsample[((cloud_downsample[:, 2] >= -1.2))] # 0.3
    cloudoutliers = cloudoutliers[((cloudoutliers[:, 2] <= 1))] # 2.5
    #cloudoutliers = cloud_downsample

    cloud_for_clustering = o3d.geometry.PointCloud()
    cloud_for_clustering.points = o3d.utility.Vector3dVector(cloudoutliers)

    # Clustering Pointcloud
    # adjust the threshold into Clustering
    labels = np.asanyarray(cloud_for_clustering.cluster_dbscan(0.7,3))
    #print("number of estimated clusters : ", len(clusters))
    #print("How much time for Clustering")
    #print(time.time() - start)

    # Visualize Clusters
    for i in range(np.max(labels)):
        # Find the Cars
        # 1) Extract each cluster
        DBSCAN_Result = cloud_for_clustering.select_by_index(np.where(labels == i)[0])
        clusterCloud = np.asarray(DBSCAN_Result.points)

        # if size of cluster <= 10, then dismiss
        if len(clusterCloud) <= 10: continue
        
        # plot cluster
        clusterCloud_plot = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ clusterCloud.T).T    
        # plt.plot(clusterCloud_plot[:,0], clusterCloud_plot[:,1],'bo', markersize = 0.4)

        # 2) Find Cars with weak condition
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])
        z_for_slicing = 6/7*z_min + 1/7*z_max

        #if 0.5 > z_for_slicing or z_for_slicing > 1: continue

        # slicing by z values
        clusterCloud = clusterCloud[(clusterCloud[:,2] >= z_for_slicing - 0.1)]#0.15
        clusterCloud = clusterCloud[(clusterCloud[:,2] <= z_for_slicing + 0.1)]
        
        # if size of cluster <= 10, then dismiss
        if len(clusterCloud) <= 10: continue

        x_max = np.max(clusterCloud[:,0])
        x_min = np.min(clusterCloud[:,0])
        y_max = np.max(clusterCloud[:,1])
        y_min = np.min(clusterCloud[:,1])        
        
        x_len = abs(x_min - x_max)
        y_len = abs(y_min - y_max)
        z_len = abs(z_min - z_max)

        # Do we need car_min condition??
        if  (carx_min < x_len or cary_min < y_len) and (x_len < carx_max and y_len < cary_max and carz_min < z_len < carz_max):
            templist_res, templist_box, flag = socar.sort_Car(clusterCloud, z_max, z_min)
            
            # Not so good cluster
            if templist_res is None:
                continue

            if(flag == True):
                cluster = clusterClass(np.array(templist_res), np.array(templist_box), i, 1)
                clusterClass_list.append(cluster)

                # print(flag)
                # plt.plot(clusterCloud_plot[:,0], clusterCloud_plot[:,1],'yo', markersize = 0.4)
                # res = np.append(res, [templist_res], axis = 0)
                # box = np.append(box, [templist_box], axis = 0)
                # car_list.append(i)
                # car_list_res.append(len(res)-1)
            else:
                cluster = clusterClass(np.array(templist_res), np.array(templist_box), i, 0)
                clusterClass_list.append(cluster)
                # res = np.append(res, [templist_res], axis = 0)
                # box = np.append(box, [templist_box], axis = 0)
        

    ########################################################################
    ############################## Tracking ################################
    ########################################################################
    #print("how many meaured?" , len(res))
    #print(res[:])

    # z_meas == res
    # z_processed = np.zeros(len(res))
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
                if Track_list[i].DelCnt >= 5:
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
            temp = cloud_for_clustering.select_by_index(np.where(labels == Track_list[i].ClusterID)[0])
            temp = np.asarray(temp.points)

            if len(temp) == 0:
                continue
            temp = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ temp.T).T
            center = np.array([Track_list[i].state[0], Track_list[i].state[1], 0])
            center = (np.array([ [0,-1,0], [1,0,0], [0,0,1]]) @ center.T).T
            # plt.plot(temp[:,0], temp[:,1], 'ro', markersize = 0.4)
            # plt.plot(center[0], center[1], 'go')
            # x, y, u, v = Track_list[i].state[0], Track_list[i].state[1], cos(Track_list[i].state[3]), sin(Track_list[i].state[3])
            # [x,y] = (np.array([ [0,-1], [1,0]]) @ np.asarray([x,y]).T).T 
            # [u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
            # plt.quiver(x, y, u, v, scale= 2, scale_units = 'inches', color = 'red')
            plt.text(center[0], center[1], 'Track{}'.format(i+1))
            
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
        Track_list_valid.append((Track_list[i],i+1))

validtracklistnum =len(Track_list_valid)
print("# of all track_list : ", len(Track_list))
print("# of valid track_list : ", validtracklistnum)

for i in range(validtracklistnum):
    index = Track_list_valid[i][1]
    start = Track_list_valid[i][0].Start
    duration = len(Track_list_valid[i][0].history_state)
    state = Track_list_valid[i][0].history_state
    framenum = frame_num
    save_to_csv(index, start, duration, state, framenum, path)
    
# reach to csv file
# csv_file = pd.read_csv('{}.csv'.format(i), index_col=0) # set i~


'''for i in range(len(Track_list)):
    print("Track_list {}-th all state".format(i+1))
    print(Track_list[i].history_state)
    plt.figure()
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,0], label = 'x_point', color = 'b')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,1], label = 'y_point', color = 'g')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,2], label = 'velocity', color = 'r')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,3], label = 'yaw-angle', color = 'c')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_state[:,4], label = 'yaw_rate', color = 'm')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_box[:,0], label = 'width', color = 'y')
    plt.plot(range(1,len(Track_list[i].history_state) + 1) , Track_list[i].history_box[:,1], label = 'length', color = 'k')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()'''
