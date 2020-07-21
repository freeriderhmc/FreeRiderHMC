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
        ###########################
        # Find the Cars
        
        # 1) Extract each cluster
        clusterCloud = cloudoutliers[clusters[i][:],:]
        
        #clustersCloud_pcd.points = np.append(clustersCloud_pcd.points,clusterCloud_pcd.points)
        
        # 2) Find Cars with weak condition
        z_max=z_min=x_max=x_min=y_max=y_min=0
        
        z_max = np.max(clusterCloud[:,2])
        z_min = np.min(clusterCloud[:,2])
        z_for_slicing = 4/5*z_min + 1/5*z_max

        '''
        #clusterCloud = clusterCloud[:,0:2]
        #clusterCloud[:,2] = z_for_slicing
        tempcloud = o3d.geometry.PointCloud()
        tempcloud.points = o3d.utility.Vector3dVector(clusterCloud)
        tempcloud.compute_convex_hull()
        
        print(type(clusterCloud))
        print(clusterCloud.shape)
        '''

        # slicing by z values
        clusterCloud = clusterCloud[(clusterCloud[:,2] >= z_for_slicing - 0.07)]#0.15
        clusterCloud = clusterCloud[(clusterCloud[:,2] <= z_for_slicing + 0.07)]
        
                
        x_max = np.max(clusterCloud[:,0])
        x_min = np.min(clusterCloud[:,0])
        y_max = np.max(clusterCloud[:,1])
        y_min = np.min(clusterCloud[:,1])
        
        
        x_len = abs(x_min - x_max)
        y_len = abs(y_min - y_max)
        z_len = abs(z_min - z_max)

        if  carx_min < x_len < carx_max and cary_min < y_len < cary_max and carz_min < z_len < carz_max:
            car_count += 1

            # Convert Numpy to Pointcloud
            clusterCloud_pcd = o3d.geometry.PointCloud()
            clusterCloud_pcd.points = o3d.utility.Vector3dVector(clusterCloud)

            convexhull = clusterCloud[(clusterCloud_pcd.compute_convex_hull()[1])[:],:]



            clusterCloud_2D = convexhull[:,0:2]
            points_x = clusterCloud_2D[:,0]
            points_y = clusterCloud_2D[:,1]


            # Line Segmentation to extract two lines
            inliers1_list, outliers1_list , _, _, _= seg.RansacLine(clusterCloud_2D, 120, 0.1)

            line1_inliers = clusterCloud_2D[inliers1_list[:], :]
            line1_outliers = clusterCloud_2D[outliers1_list[:], :]
            

            #outliers = clusterCloud_2D[outliers1_list[:],:]
            inliers2_list, outliers2_list ,_,_,_= seg.RansacLine(line1_outliers, 60, 0.2)

            line2_inliers = line1_outliers[inliers2_list[:],:]

            ############################################################################
            #######################################Linear Regression ###################
            ############################################################################
            line_fitter1 = LinearRegression()
            line_fitter2 = LinearRegression()
            len1 = len(line1_inliers[:][:,0])
            len2 = len(line2_inliers[:][:,0])

            xline1 = line1_inliers[:][:,0].reshape(len1,1)
            yline1 = line1_inliers[:][:,1].reshape(len1,1)
            xline2 = line2_inliers[:][:,0].reshape(len2,1)
            yline2 = line2_inliers[:][:,1].reshape(len2,1)

            line1_fit = line_fitter1.fit(xline1,yline1)
            line2_fit = line_fitter2.fit(xline2,yline2)
            line1dy = line1_fit.coef_
            # line1bias = line1_fit.intercept_
            line1pred = line1_fit.predict(xline1).reshape([len1,1])
            
            line2dy = line2_fit.coef_
            line2pred = line2_fit.predict(xline2).reshape([len2,1])

            line1dict = {}
            line2dict = {}
            for i in range(0,len1):
                line1dict[line1_inliers[i][0]] = line1_inliers[i][:]
            
            for i in range(0,len2):
                line2dict[line2_inliers[i][0]] = line2_inliers[i][:]

            line1dict_sorted = sorted(line1dict.items())
            line2dict_sorted = sorted(line2dict.items())

            len1 = len(line1dict_sorted)
            len2 = len(line2dict_sorted)

            line1_sorted = np.empty([0,2])
            line2_sorted = np.empty([0,2])
            
            
            for j in range(0,len1):
                line1_sorted = np.append(line1_sorted, [line1dict_sorted[j][1]],axis = 0)

            for j in range(0, len2):
                line2_sorted = np.append(line2_sorted, [line2dict_sorted[j][1]],axis = 0) 

            x1, y1 = line1_sorted[0][0], line1_sorted[0][1]
            x2, y2 = line1_sorted[len1-1][0], line1_sorted[len1-1][1]
            x3, y3 = line2_sorted[0][0], line2_sorted[0][1]

            x1x3 = ((x1-x3)**2+(y1-y3)**2)**0.5
            x2x3 = ((x2-x3)**2+(y2-y3)**2)**0.5

            if(x1x3 < 0.4):
                x3, y3 = line2_sorted[len2-1][0], line2_sorted[len2-1][1]
                delx, dely = x3-x1, y3-y1
                x4 =  x2+delx
                y4 =  y2+dely
                centroid_x = (x1+x2+x3+x4)/4
                centroid_y = (y1+y2+y3+y4)/4
                w = x1x3

            elif(x2x3 < 0.4):
                x3, y3 = line2_sorted[len2-1][0], line2_sorted[len2-1][1]
                delx, dely = x3-x2, y3-y2
                x4 =  x1+delx
                y4 =  y1+dely
                centroid_x = (x1+x2+x3+x4)/4
                centroid_y = (y1+y2+y3+y4)/4
                w = x2x3

            else:
                if(x1x3 < x2x3):
                    centroid_x = (x2+x3)/2
                    centroid_y = (y2+y3)/2
                    w = x1x3
                else:
                    centroid_x = (x1+x3)/2
                    centroid_y = (y1+y3)/2
                    w = x2x3

            yaw = get_angle([1,line1dy])
            l = (abs(x1-x2)**2+abs(y1-y2)**2)**0.5
            h = z_max - z_min + 0.5

            if(l<w):
                temp = w
                w = l
                l = temp
                yaw = get_angle([1,line2dy])

            ang1 = get_angle([1, line1dy])*180/pi
            ang2 = get_angle([1, line2dy])*180/pi
            print(abs(ang1-ang2))
            if(65<abs(ang1-ang2)<110 ):
                res = np.append(res, [[centroid_x,centroid_y, yaw, w, l, h]], axis = 0)

                # plt.figure()
                # plt.plot(points_x, points_y, 'g*')
                # plt.plot(line1_inliers[:,0],line1_inliers[:,1], 'o')
                # plt.plot(line2_inliers[:,0],line2_inliers[:,1], 'ro')
                # plt.scatter(centroid_x,centroid_y,color ='blue')
                # plt.show()

            '''
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(clusterCloud)

            if i%3 == 0:
                temp_pcd.paint_uniform_color([1,0,0])
            elif i%3 == 1:
                temp_pcd.paint_uniform_color([0,1,0])
            else:
                temp_pcd.paint_uniform_color([0,0,1])

            vis.add_geometry(temp_pcd)
            vis.run()
            '''  
    num += 1
    
    input("Press Enter to continue...")

    #vis.clear_geometries()   
#vis.destroy_window()
