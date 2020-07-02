# FreeRiderHMC Team
# Ver.3    0702
# Segment the road just by cutting the z values below threshold instead of RANSAC segmentation
# Visualize Each Clusters
# load binary data
# Add sequent visualization

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



vis = o3d.visualization.Visualizer()
vis.create_window()


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
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] <= 10))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] >= -10))]

    # threshold z value cut the road
    cloudoutliers = cloud_downsample[((cloud_downsample[:, 2] >= -1.4))] # -1.56
    print(len(cloudoutliers))


    # Clustering Pointcloud
    # adjust the threshold into Clustering
    start = time.time()

    tree = KDTree(cloudoutliers)
    clusters = clu.euclideanCluster(cloudoutliers, tree, 0.35)
    print("number of estimated clusters : ", len(clusters))
    
    print("How much time for Clustering")
    print(time.time() - start)

    

    clustersCloud_pcd = o3d.geometry.PointCloud()
    # Visualize Clusters
    for i in range(len(clusters)):
        #print(len(clusters[i]))
        clusterCloud = cloudoutliers[clusters[i][:],:]
        clusterCloud_pcd = o3d.geometry.PointCloud()
        clusterCloud_pcd.points = o3d.utility.Vector3dVector(clusterCloud)
        if i%3 == 0:
            clusterCloud_pcd.paint_uniform_color([1,0,0])
        elif i%3 == 1:
            clusterCloud_pcd.paint_uniform_color([0,1,0])
        else:
            clusterCloud_pcd.paint_uniform_color([0,0,1])

        #clustersCloud_pcd.points = np.append(clustersCloud_pcd.points,clusterCloud_pcd.points)
        
        vis.add_geometry(clusterCloud_pcd)
        vis.run()
        
        #o3d.visualization.draw_geometries([clusterCloud_pcd])
        time.sleep(0.5)

    # enter key -> next frame
    input("Press Enter to continue...")
    vis.clear_geometries()
    
        
        
        
    
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