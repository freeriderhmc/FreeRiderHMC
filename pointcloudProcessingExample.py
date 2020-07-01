# FreeRiderHMC Team
# Ver.2    0701
# Segment the road just by cutting the z values below threshold instead of RANSAC segmentation
# Visualize Each Clusters
# load binary data

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


# Load binary data
path = './2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/'

file_list = loadData.load_data(path)

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

    # Convert pcd to numpy array
    cloud_downsample = np.asarray(cloud_downsample.points)

    # Crop Pointcloud -20m < x < 20m && -20m < y < 20m && z > -1.80m
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] <= 20))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] >= -20))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] <= 20))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] >= -20))]

    # threshold z value cut the road
    cloudoutliers = cloud_downsample[((cloud_downsample[:, 2] >= -1.4))] # -1.56
    print(len(cloudoutliers))


    # Clustering Pointcloud
    # adjust the threshold into Clustering
    start = time.time()

    tree = KDTree(cloudoutliers)
    clusters = clu.euclideanCluster(cloudoutliers, tree, 0.15)
    print("number of estimated clusters : ", len(clusters))
    print("How much time for Clustering")
    print(time.time() - start)

    '''
    # Visualize Clusters
    for i in range(len(clusters)):
        clusterCloud = np.empty(shape=[0, 3])
        for j in range(len(clusters[i])):
            clusterCloud = np.append(clusterCloud, np.array([cloudoutliers[clusters[i][j]]]), axis = 0)
            # numpy -> pcd
            # using o3d.utility.color~
            # update geometry 
    '''
        



    # Visualization
    pcd_processed = o3d.geometry.PointCloud()
    pcd_processed.points = o3d.utility.Vector3dVector(cloudoutliers)

    o3d.visualization.draw_geometries([pcd_processed])
    time.sleep(5)



'''
# Load a pcd data
pcd_load = o3d.io.read_point_cloud("./kitti_pcd/pcd0000000000.pcd")
pcd_downsample = pcd_load.voxel_down_sample(voxel_size=0.3)
print(pcd_load)
'''