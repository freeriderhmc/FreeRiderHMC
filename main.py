import sys
import os
import numpy as np
import open3d as o3d
import time
import math
import operator
from sklearn.cluster import MeanShift
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import clusteringModule as clu
import lineSegmentation as seg
import loadData
import sortCar as socar

####################################################
########### Setting ################################
####################################################

pi = 3.141592653589793238

def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    return angle


# Set mod
mod = sys.modules[__name__]

# Expand iteration limit
sys.setrecursionlimit(5000)

# Set Car Standard
carz_min, carz_max = 0, 2
carx_min, carx_max = 1.5, 5
cary_min, cary_max = 1.5, 5

# Set Visualizer and Draw x, y Axis
#vis = o3d.visualization.Visualizer()
#vis.create_window()

Axis_Points = [[0,0,0], [20,0,0],[0,20,0]]
Axis_Lines = [[0,1],[0,2]]

colors = [[0,0,0] for i in range(len(Axis_Lines))]

line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(Axis_Points), lines = o3d.utility.Vector2iVector(Axis_Lines))
line_set.colors = o3d.utility.Vector3dVector(colors)


# Load binary data
path = '../data/'



file_list = loadData.load_data(path)
num = 0
car_count = 0

##################################################################################
########################### Main Loop ############################################
##################################################################################


# get points from all lists
for files in file_list:
    res = np.empty([0,6])
    # Draw Axis
    #vis.add_geometry(line_set)
    #vis.run()

    data = np.fromfile(path+files, dtype = np.float32)
    data = data.reshape(-1,4)
    data = data[:,0:3]

    # Convert numpy into pointcloud 
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)

    # Downsampling pointcloud
    cloud_downsample = cloud.voxel_down_sample(voxel_size=0.1)

    #print(cloud_downsample.segment_plane(0.4,300,300)[1])
    #outerBox = [[20,-10,-1.8],[20,-10,-1.8]]
    #cloud_downsample.crop()


    # Convert pcd to numpy array
    cloud_downsample = np.asarray(cloud_downsample.points)

    # Crop Pointcloud -20m < x < 20m && -20m < y < 20m && z > -1.80m
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] <= 15))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] >= -15))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] <= 10))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] >= -10))]

    # threshold z value cut the road
    cloudoutliers = cloud_downsample[((cloud_downsample[:, 2] >= -1.3))] # -1.56


    # Clustering Pointcloud
    # adjust the threshold into Clustering
    start = time.time()

    tree = KDTree(cloudoutliers, leaf_size = 100)
    clusters = clu.euclideanCluster(cloudoutliers, tree, 0.5)
    #print("number of estimated clusters : ", len(clusters))
    #print("How much time for Clustering")
    #print(time.time() - start)

    cluster = np.empty([0,3])

    clustersCloud = np.empty(shape = [0,3])
    # Visualize Clusters
    for i in range(len(clusters)):

        car_count = 0
        # Find the Cars
        # 1) Extract each cluster
        clusterCloud = cloudoutliers[clusters[i][:],:]
        
        # 2) Find Cars with weak condition
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])
        z_for_slicing = 4/5*z_min + 1/5*z_max

        # slicing by z values
        clusterCloud = clusterCloud[(clusterCloud[:,2] >= z_for_slicing - 0.08)]#0.15
        clusterCloud = clusterCloud[(clusterCloud[:,2] <= z_for_slicing + 0.08)]
        
                
        x_max = np.max(clusterCloud[:,0])
        x_min = np.min(clusterCloud[:,0])
        y_max = np.max(clusterCloud[:,1])
        y_min = np.min(clusterCloud[:,1])
        
        
        x_len = abs(x_min - x_max)
        y_len = abs(y_min - y_max)
        z_len = abs(z_min - z_max)

        if  carx_min < x_len < carx_max and cary_min < y_len < cary_max and carz_min < z_len < carz_max:
            templist = socar.sort_Car(clusterCloud, z_max, z_min)
            if(templist is not None):
                res = np.append(res, [templist], axis = 0)
                
    num += 1
    
    input("Press Enter to continue...")
