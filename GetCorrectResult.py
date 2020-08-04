import numpy as np
import math

def get_angle_2pi(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    if input_list[1]<0:
        angle = angle+2*pi
    return angle*180/pi

def getPoint_point(point,box,flag):
    w,l = box[0],box[1]
    x1,x2,x3 = point[0],point[1],point[2]
    trackid = track

    if flag == 1:
        temp = ((x2-x3-x1)^2+(y2-y3-y1)^2)**0.5
        if abs(temp-w)<abs(temp-l):
            if y2<0: l = -l
            x3, y3 = x2, y2+l
            x4, y4 = x1, y1+l
            l = abs(l)
        else:
            if x2<0: w = -w
            x3, y3 = x2+w, y2
            x4, y4 = x1+w, y1
            w = abs(w)

    elif flag == 2:
        temp_w = ((x2-x1)^2+(y2-y1)^2)**0.5
        temp_l = ((x2-x3)^2+(y2-y3)^2)**0.5

        vector_w = [x1-x2,y1-y2]
        vector_l = [x3-x2,y3-y2]

        x1 = vector_w * (w/temp_w) + x2
        y1 = vector_w * (w/temp_w) + y2
        
        x3 = vector_l * (w/temp_l) + x2
        y3 = vector_l * (w/temp_l) + y2
        
        x4 = x3 + (x1-x2) 
        y4 = y3 + (y1-y2) 

    center = [(x1+x2+x3+x4)/4,(y1+y2+y3+y4)/4]
    point = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    return center,point
    

def getHeading(point,center_his):
# def getHeading(linedy,center_his):    
    vel = center_his[0] - center_his[1]
    delta = point[2]-point[1]
    heading = get_angle_2pi(vel)

    if vel[0] < 0:
        yaw = get_angle_2pi(delta)
        #yaw = get_angle_2pi([1,linedy])
    return heading, yaw



