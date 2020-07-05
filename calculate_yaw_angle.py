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

import math

mod = sys.modules[__name__]

vis = o3d.visualization.Visualizer()
#vis.create_window()


# Load binary data
path = '../data/'

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
    clusters = clu.euclideanCluster(cloudoutliers, tree, 0.5)
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
        
        convexhull = clusterCloud[(clusterCloud_pcd.compute_convex_hull()[1])[:],:]
        length = len(convexhull)
        
        for j in range(length):
            #x_0~x_len-1
            setattr(mod, 'x_{}'.format(j), convexhull[j][0])
            setattr(mod, 'y_{}'.format(j), convexhull[j][1])

        ybias = np.array([])
        d = np.array([])
        dy = np.array([])
        target = np.array([])

        for k in range(length):
            # from x(0),, iterate
            if(k<length-1):
                #until dy(n-1) = yn-y(n-1)/xn-x(n-1)
                dy = np.append(dy, np.array([(getattr(mod, 'y_{}'.format(k+1)) - getattr(mod, 'y_{}'.format(k)))/(getattr(mod, 'x_{}'.format(k+1)) - getattr(mod, 'x_{}'.format(k)))]))
            else:
                # dy(n) = y0-yn/x0-xn
                dy = np.append(dy, np.array([(y_0 - getattr(mod, 'y_{}'.format(length-1)))/(x_0 - getattr(mod, 'x_{}'.format(length-1)))]))
            for l in range(length):
                # calculate ybias at one dy
                
                ybias = np.append(ybias, np.array([getattr(mod, 'y_{}'.format(l)) - dy[k]*getattr(mod, 'x_{}'.format(l))])) # so many ybias
                # calculate maximum distance
                target = np.append(target, np.array([abs(ybias[l] - (getattr(mod, 'y_{}'.format(k))-dy[k]*getattr(mod, 'x_{}'.format(k))))/(dy[k]**2+1)**0.5]))
            #max_idx = np.where(ybias == np.max(ybias))[0][0]
            #min_idx = np.where(ybias == np.min(ybias))[0][0]
            d = np.append(d, np.array([np.max(target)]))
            target = np.array([])
            ybias = np.array([])
        
        max_box = np.where(d==np.max(d))[0][0]
        print(math.atan(dy[max_box])*180/math.pi)
        '''
        vert_dy = -(1/dy[max_box])
        vert_ybias = np.array([])
        y = getattr(mod, 'y_{}'.format(max_box))
        x = getattr(mod, 'x_{}'.format(max_box))
        for m in range(length):
            vert_ybias = np.append(vert_ybias, np.array([getattr(mod, 'y_{}'.format(m)) - vert_dy* getattr(mod, 'x_{}'.format(m))]))
        minvert_ybias = np.min(vert_ybias)
        maxvert_ybias = np.max(vert_ybias)
        '''

        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(convexhull)
        if i%3 == 0:
            pcd.paint_uniform_color([1,0,0])
        elif i%3 == 1:
            pcd.paint_uniform_color([0,1,0])
        else:
            pcd.paint_uniform_color([0,0,1])

        #clustersCloud_pcd.points = np.append(clustersCloud_pcd.points,clusterCloud_pcd.points)
        '''
        #vis.add_geometry(clusterCloud_pcd)
        #vis.run()
        
        #o3d.visualization.draw_geometries([clusterCloud_pcd])
        #time.sleep(0.5)

    # enter key -> next frame
    input("Press Enter to continue...")
    vis.clear_geometries()
