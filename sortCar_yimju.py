import numpy as np
import open3d as o3d
import math
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


import lineSegmentation as seg
# import sortline as sl

############################## Macro ###############################
pi = 3.141592653589793238

######################### Define Function ##########################
def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    # if angle<-pi/4:
    #     angle = angle + pi
    
    return angle

def get_distance(xy1,xy2):
    distance = ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5
    return distance

def sortline_co(line):
    length = len(line[:][:,0])
    linedict = {}

    for i in range(0,length):
        linedict[line[:][i,0]] = line[:][i,:]

    linedict_sorted = sorted(linedict.items())
    line_sorted = np.empty([0,2])
    length = len(linedict_sorted)

    for j in range(0,length):
        line_sorted = np.append(line_sorted, [linedict_sorted[j][1]],axis = 0)

    return line_sorted


def sortline_angle(line, inner_point):
    length = len(line[:][:,0])
    linedict = {}
    linevectors = line - inner_point

    listangle = list(map(get_angle, linevectors))
    
    for i in range(0,length):
        #line1dict[xline1[i]] = [xline1[i],yline1[i]]
        linedict[listangle[i]] = line[:][i,:]
    linedict_sorted = sorted(linedict.items())

    listangle = sorted(listangle)
    
    line_sorted = np.empty([0,2])

    length = len(linedict_sorted)
    
    for j in range(0,length):
        line_sorted = np.append(line_sorted, [linedict_sorted[j][1]],axis = 0)

    for i in range(0,length-1):
        theta = abs(listangle[i]-listangle[i+1])
        if 180 < theta:            
            move = line_sorted[:i+1]
            line_sorted = line_sorted[i+1:]
            line_sorted = np.append(line_sorted,move,axis = 0)

    return line_sorted



##########################################################################
############################# Main Function ##############################
##########################################################################


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
    
    tmp1 = seg.RansacLine(clusterCloud_2D, 90, 0.1)
    if(tmp1 is not None):
        inliers1_list, outliers1_list = tmp1
    else:
        return None, None, None

    if(len(inliers1_list)==0 or len(outliers1_list)==0):
        return None, None, None

    line1_inliers = clusterCloud_2D[inliers1_list[:], :]
    line1_outliers = clusterCloud_2D[outliers1_list[:], :]

    if(len(line1_outliers)==0):
        line_fitter1 = LinearRegression()
        xline1 = line1_inliers[:][:,0].reshape(len1,1)
        yline1 = line1_inliers[:][:,1].reshape(len1,1)
        line1_fit = line_fitter1.fit(xline1,yline1)
        line1dy = line1_fit.coef_

        line1_sorted = sortline_co(line1_inliers)
        len1 = len(line1_inliers[:][:,0])
        x1, y1 = line1_sorted[0][0], line1_sorted[0][1]
        x2, y2 = line1_sorted[len1-1][0], line1_sorted[len1-1][1]

        center = [(x1+x2)/2, (y1+y2)/2]
        yaw = get_angle([1,line1dy])
        dis_temp = ((x1-x2)**2+(y1-y2)**2)**0.5
        flag = True
        w,l = 0
        print('line1')

        if 1 < dis_temp <6:
            print(flag)
            l = dis_temp                       
            return [center[0], center[1], yaw], [w, l,h], flag
        elif  1 < dis_temp <2.8:
            w = dis_temp
            return [center[0], center[1], yaw+pi/2], [w, l,h], flag

        else: return None, None, None

    else:                 
        tmp = seg.RansacLine(line1_outliers, 45, 0.1)
        
        if(tmp is not None):
            inliers2_list, _ = tmp
        else:
            return None, None, None

        line2_inliers = line1_outliers[inliers2_list[:],:]

        ####################### Linear Regression ######################

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

        line1_sorted = sortline_angle(line1_inliers,inner_point)
        line2_sorted = sortline_angle(line2_inliers,inner_point)
        
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

        line1_sorted_plot = (np.array([ [0,-1], [1,0]]) @ line1_sorted.T).T    
        line2_sorted_plot = (np.array([ [0,-1], [1,0]]) @ line2_sorted.T).T    
        center_plot = (np.array([ [0,-1], [1,0]]) @ np.asarray(center).T).T    
        plt.plot(line1_sorted_plot[:,0],line1_sorted_plot[:,1], 'bo', markersize = 0.8)
        plt.plot(line2_sorted_plot[:,0],line2_sorted_plot[:,1], 'ro', markersize = 0.8)
        plt.scatter(center_plot[0],center_plot[1],color ='green')

        # if -> Car
        # else -> Not Car but cluster
        #if(62<abs(ang1-ang2)<131.2): flag = True
        if(50<abs(ang1-ang2)<131.2): 
            flag = True


            
            #plt.show()      
        else: flag = False
            #return None, None, None

    return [center[0], center[1], yaw], [w, l,h], flag

if __name__ == "__main__":
    print("Error.. Why sortCar Module execute")