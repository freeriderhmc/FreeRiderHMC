import numpy as np
import open3d as o3d
import math
from math import cos, sin, pi
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


import lineSegmentation as seg
# import sortline as sl

############################## Macro ###############################
# pi = 3.141592653589793238

######################### Define Function ##########################

def get_dy2yaw(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    if angle<-pi/4:
        angle = angle + pi
    return angle

def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    if input_list[1]<0:
        angle = angle+2*pi
    return angle*180/pi

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


def sort_Car(points, center):
    
    ####################### Get result #########################            

    points_sorted = sortline_angle(points, center)
    x1, y1 = points_sorted[0][0], points_sorted[0][1]
    x2, y2 = points_sorted[1][0], points_sorted[1][1]
    x3, y3 = points_sorted[2][0], points_sorted[2][1]
    x4, y4 = points_sorted[3][0], points_sorted[3][1]
   
    x1x2 = ((x1-x3)**2+(y1-y3)**2)**0.5
    x2x3 = ((x2-x3)**2+(y2-y3)**2)**0.5
    ang12 = get_dy2yaw([x1-x2,y1-y2])
    ang32 = get_dy2yaw([x3-x2,y3-y2])
    w = x1x2
    l = x2x3
    
    if x1x2 < x2x3:
        w = x2x3
        l = x1x2
        yaw = ang32    
    

    if abs(center[1])<2 and (w<1 or l<3):
        temp = w
        w = l
        l = temp
        if len(line1_sorted)>=len(line2_sorted):
            yaw = ang12
        else: yaw = ang32
        if yaw > pi/4: yaw = yaw - pi/2            

    ang1 = ang12*180/pi
    ang2 = ang32*180/pi  
    # if -> Car
    # else -> Not Car but cluster
    #if(62<abs(ang1-ang2)<131.2): flag = True
    if(50<abs(ang1-ang2)<131.2): pass
    else: flag = False
        #return None, None, None



    line1_sorted_plot = (np.array([ [0,-1], [1,0]]) @ line1_sorted.T).T    
    line2_sorted_plot = (np.array([ [0,-1], [1,0]]) @ line2_sorted.T).T    
    center_plot = (np.array([ [0,-1], [1,0]]) @ np.asarray(center).T).T   
    # plt.figure() 
    plt.plot(line1_sorted_plot[:,0],line1_sorted_plot[:,1], 'bo', markersize = 0.8)
    plt.plot(line2_sorted_plot[:,0],line2_sorted_plot[:,1], 'ro', markersize = 0.8)
    x, y, u, v = point[1][0], point[1][1], cos(yaw), sin(yaw)
    [x,y] = (np.array([ [0,-1], [1,0]]) @ np.asarray([x,y]).T).T 
    [u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
    plt.quiver(x, y, u, v, scale= 2, scale_units = 'inches', color = 'red')
    # plt.show()
    return [center[0], center[1], yaw], [w, l], flag


    # return [center[0], center[1], yaw, point], [w, l,h], flag

if __name__ == "__main__":
    print("Error.. Why sortCar Module execute")
