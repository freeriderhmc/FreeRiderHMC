import numpy as np
import open3d as o3d
import math
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

import lineSegmentation as seg

############################## Macro ###############################
pi = 3.141592653589793238

######################### Define Function ##########################
def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    return angle

def get_distance(xy1,xy2):
    distance = ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5
    return distance

def sort_Car(clusterCloud, z_max, z_min):

    # Get Centroid
    x_sum = clusterCloud[:,0].sum() 
    y_sum = clusterCloud[:,1].sum()

    num_points = len(clusterCloud)
    x_inner = x_sum / num_points
    y_inner = y_sum / num_points

    inner_point = [x_inner, y_inner]
    if (len(inner_point)==0) : return 0,0,0

    # Convert Numpy to Pointcloud
    clusterCloud_pcd = o3d.geometry.PointCloud()
    clusterCloud_pcd.points = o3d.utility.Vector3dVector(clusterCloud)

    convexhull = clusterCloud[(clusterCloud_pcd.compute_convex_hull()[1])[:],:]

    clusterCloud_2D = convexhull[:,0:2]
    #points_x = clusterCloud_2D[:,0]
    #points_y = clusterCloud_2D[:,1]

    # Line Segmentation to extract two lines
    
    tmp1 = seg.RansacLine(clusterCloud_2D, 140, 0.1)
    if(tmp1 is not None):
        inliers1_list, outliers1_list = tmp1
    else:
        return None, None, None

    if(len(inliers1_list)==0 or len(outliers1_list)==0):
        return None, None, None

    line1_inliers = clusterCloud_2D[inliers1_list[:], :]
    line1_outliers = clusterCloud_2D[outliers1_list[:], :]
    if(len(line1_outliers)==0):
        return None, None, None

    tmp = seg.RansacLine(line1_outliers, 70, 0.2)
    
    if(tmp is not None):
        inliers2_list, _ = tmp
    else:
        return None, None, None

    line2_inliers = line1_outliers[inliers2_list[:],:]

    #######################################Linear Regression ###################
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
    #line1pred = line1_fit.predict(xline1).reshape([len1,1])

    line2dy = line2_fit.coef_
    #line2pred = line2_fit.predict(xline2).reshape([len2,1])

    line1dict = {}
    line2dict = {}

    line1vectors = line1_inliers - inner_point
    line2vectors = line2_inliers - inner_point

    list1angle = list(map(get_angle, line1vectors))
    list2angle = list(map(get_angle, line2vectors))
    
    for i in range(0,len1):
        line1dict[line1_inliers[i][0]] = line1_inliers[i][:]
    for i in range(0,len2):
        line2dict[line2_inliers[i][0]] = line2_inliers[i][:]

    line1dict_sorted = sorted(line1dict.items())
    line2dict_sorted = sorted(line2dict.items())
    list1angle = sorted(list1angle)
    list2angle = sorted(list2angle)

    line1_sorted = np.empty([0,2])
    line2_sorted = np.empty([0,2])

    len1 = len(line1dict_sorted)
    len2 = len(line2dict_sorted)

    for j in range(0,len1):
        line1_sorted = np.append(line1_sorted, [line1dict_sorted[j][1]],axis = 0)

    for j in range(0, len2):
        line2_sorted = np.append(line2_sorted, [line2dict_sorted[j][1]],axis = 0) 



    for i in range(0,len1-1):
        theta = abs(list1angle[i]-list1angle[i+1])
        if 180 < theta:            
            move = line1_sorted[:i+1]
            line1_sorted = line1_sorted[i+1:]
            line1_sorted = np.append(line1_sorted,move,axis = 0)

    for i in range(0,len2-1):
        theta = abs(list2angle[i]-list2angle[i+1])
        if 180<theta:                
            move = line2_sorted[:i+1 ]
            line2_sorted = line2_sorted[i+1 :]
            line2_sorted = np.append(line2_sorted,move,axis = 0)

    ####################### Get result #########################            
    x1, y1 = line1_sorted[0][0], line1_sorted[0][1]
    x2, y2 = line1_sorted[len1-1][0], line1_sorted[len1-1][1]
    x3, y3 = line2_sorted[0][0], line2_sorted[0][1]
    x4, y4 = line2_sorted[len2-1][0], line2_sorted[len2-1][1]

    x1x3 = ((x1-x3)**2+(y1-y3)**2)**0.5
    x2x3 = ((x2-x3)**2+(y2-y3)**2)**0.5
    x1x4 = ((x1-x4)**2+(y1-y4)**2)**0.5
    x2x4 = ((x2-x4)**2+(y2-y4)**2)**0.5
    w = ((x3-x4)**2+(y3-y4)**2)**0.5

    delx = x2-x1
    dely = y2-y1

    if(x2x3<x1x3):
        if(x2x4<x2x3):
            x4 = x3-delx
            y4 = y3-dely
        else:
            x3 = x4-delx
            y3 = y4-dely

    else:
        if(x1x4<x1x3):
            x4 = x3+delx
            y4 = y3+dely

        else:
            x3 = x4+delx
            y3 = y4+dely

    center = [(x1+x2+x3+x4)/4,(y1+y2+y3+y4)/4]
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

    # if -> Car
    # else -> Not Car but cluster
    if(62<abs(ang1-ang2)<131.2):
        return [center[0], center[1], yaw], [w, l,h], True
    else:
        return [center[0], center[1], yaw], [w, l,h], False


if __name__ == "__main__":
    print("Error.. Why sortCar Module execute")