# FreeRiderHMC Team
# main_Ver.2    0703
# Segment the road just by cutting the z values below threshold instead of RANSAC segmentation
# Visualize Each Clusters
# load binary data
# Add sequent visualization
# Add z slicing

import sys
import os
import numpy as np
import open3d as o3d
import time
from sklearn.cluster import MeanShift
from sklearn.neighbors import KDTree
import clusteringModule as clu
import planeSegmentation as seg
import loadData

####################################################
########### Setting ################################
####################################################

# Expand iteration limit
sys.setrecursionlimit(5000)

# Set Car Standard
carz_min, carz_max = 0, 2
carx_min, carx_max = 1.5, 5
cary_min, cary_max = 1.5, 5

# Set Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Load binary data
path = './2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/'

file_list = loadData.load_data(path)
num = 0
car_count = 0

##################################################################################
########################### Main Loop ############################################
##################################################################################


# get points from all lists
for files in file_list:
    data = np.fromfile(path+files, dtype = np.float32)
    data = data.reshape(-1,4)
    data = data[:,0:3]

    # Convert numpy into pointcloud 
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)

    # Downsampling pointcloud
    cloud_downsample = cloud.voxel_down_sample(voxel_size=0.2)

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
    clusters = clu.euclideanCluster(cloudoutliers, tree, 0.6)
    print("number of estimated clusters : ", len(clusters))
    
    print("How much time for Clustering")
    print(time.time() - start)

    cluster = np.empty([1,3])

    clustersCloud = np.empty(shape = [0,3])
    # Visualize Clusters
    for i in range(len(clusters)):
        car_count = 0
        ###########################
        # Find the Cars
        
        # 1) Extract each cluster
        clusterCloud = cloudoutliers[clusters[i][:],:]
        
        #clustersCloud_pcd.points = np.append(clustersCloud_pcd.points,clusterCloud_pcd.points)
        
        # 2) Find Cars with weak condition
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])
        z_for_slicing = 2/3*z_min + 1/3*z_max


        # slicing by z values
        clusterCloud[(clusterCloud[:,2] >= z_for_slicing - 0.3)]
        clusterCloud[(clusterCloud[:,2] <= z_for_slicing + 0.3)]
                
        x_max = np.max(clusterCloud[:,0])
        x_min = np.min(clusterCloud[:,0])
        y_max = np.max(clusterCloud[:,1])
        y_min = np.min(clusterCloud[:,1])
        
        
        x_len = abs(x_min - x_max)
        y_len = abs(y_min - y_max)
        z_len = abs(z_min - z_max)

        if  carx_min < x_len < carx_max and cary_min < y_len < cary_max and carz_min < z_len < carz_max:
            car_count += 1


            ###########################################################
            ###########  convex hull -> width, length, yaw angle ######
            ###########################################################
            ############ after convex hull , if not a car -> continue (next loop)


            clusterCloud_pcd = o3d.geometry.PointCloud()
            clusterCloud_pcd.points = o3d.utility.Vector3dVector(clusterCloud)
            
            # For Painting if it is a car
            if i%3 == 0:
                clusterCloud_pcd.paint_uniform_color([1,0,0])
            elif i%3 == 1:
                clusterCloud_pcd.paint_uniform_color([0,1,0])
            else:
                clusterCloud_pcd.paint_uniform_color([0,0,1])

            # Visualization
            vis.add_geometry(clusterCloud_pcd)
            vis.run()

    print(car_count)
    num += 1
    print (num)
    

    # if want to pause at each frame 
    #input("Press Enter to continue...")

    vis.clear_geometries()
    
        
        
vis.destroy_window()
        
    
    #clusterCloud.paint_uniform_color([0.1, 0.9, 0.1])



    # Visualization
    #pcd_processed = o3d.geometry.PointCloud()
    #pcd_processed.points = o3d.utility.Vector3dVector(cloudoutliers)

    



'''
# Load a pcd data
pcd_load = o3d.io.read_point_cloud("./kitti_pcd/pcd0000000000.pcd")
pcd_downsample = pcd_load.voxel_down_sample(voxel_size=0.3)
print(pcd_load)
'''