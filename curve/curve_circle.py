import numpy as np
from math import atan, acos, pi
# from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import circle_fit as cf

# prev_leftdy = 0
# prev_rightdy = 0
# prev_leftc = 0
# prev_rightc = 0

def curve(left_lane, right_lane):    

    left_lane = left_lane[left_lane[:,0]<20]
    right_lane = right_lane[right_lane[:,0]<20]

    xleft_plot = np.arange(5,40,0.01).reshape(-1,1)
    xright_plot = np.arange(5,40,0.01).reshape(-1,1)

    if len(left_lane) < 4 and len(right_lane) < 4:
        leftdy, leftc = 0,1.5
        rightdy, rightc = 0,-1.5
        yleft_plot = np.ones(3500)*1.5
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(3500)*(-1.5)
        yright_plot = yright_plot.reshape(-1,1)

        flag = False
        left_fit = [flag, leftdy,leftc]
        right_fit = [flag, rightdy,rightc]

    ########################### Only Right ############################
    elif len(left_lane)< 4:
        xc,yc,right_r,_ = cf.least_squares_circle(right_lane)
        # left_center = direction*left_r +1.5       
        if yc<0: direction = -1
        else: direction = 1  
        right_center = [xc,yc]


        if 20 <right_r:
            yright_plot = circle_plot(xright_plot,right_center,right_r)
            yleft_plot = yright_plot+3                        

            left_r = right_r-direction*3
            left_center = right_center

            left_fit = [left_r,left_center, direction]
            right_fit = [right_r,right_center, direction]

        else:
            yleft_plot = np.ones(3500)*2
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-2)
            yright_plot = yright_plot.reshape(-1,1)

            leftdy, leftc = 0,2
            rightdy, rightc = 0,-2

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]
           
    ############################ Only Left ############################

    elif len(right_lane)<4:
        xc,yc,left_r,_ = cf.least_squares_circle(left_lane)
        # left_center = direction*left_r +1.5       
        if yc<0: direction = -1
        else: direction = 1  
        left_center = [xc,yc]
        
        if 20 <left_r:
            yleft_plot = circle_plot(xleft_plot,left_center,left_r)
            yright_plot = yleft_plot+3                        

            right_r = left_r-direction*3
            right_center = left_center

            left_fit = [left_r,left_center, direction]
            right_fit = [right_r,right_center, direction]
        else:
            yleft_plot = np.ones(3500)*1.5
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-1.5)
            yright_plot = yright_plot.reshape(-1,1)

            leftdy, leftc = 0,1.5
            rightdy, rightc = 0,-1.5

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]

    ############################ Both Lanes ############################

    else:
        xc,yc,left_r,_ = cf.least_squares_circle(left_lane)
        # left_center = direction*left_r +1.5       
        if yc<0: left_direction = -1
        else: left_direction = 1  
        left_center = [xc,yc]
        

        xc,yc,right_r,_ = cf.least_squares_circle(right_lane)
        # left_center = direction*left_r +1.5       
        if yc<0: right_direction = -1
        else: right_direction = 1  
        right_center = [xc,yc]

        same = left_direction*right_direction
        

        if 20 <left_r and 20 <right_r and same >0 and abs(left_r-right_r)<5:
            direction = left_direction
            yleft_plot = circle_plot(xleft_plot,left_center,left_r)
            yright_plot = circle_plot(xright_plot,right_center,right_r)
           
            left_fit = [left_r,left_center, direction]
            right_fit = [right_r,right_center, direction]

        elif 20 <left_r and same>0:
            direction = left_direction
            yleft_plot = circle_plot(xleft_plot,left_center,left_r)            
            yright_plot = yleft_plot+3       
            left_fit = [left_r,left_center, direction]
            right_fit = [right_r,right_center, direction]

        elif 20 <right_r and same>0:
            direction = left_direction
            yright_plot = circle_plot(xright_plot,right_center,right_r)
            yleft_plot = yright_plot+3                       

            left_fit = [left_r,left_center, direction]
            right_fit = [right_r,right_center, direction]           
        else:
            yleft_plot = np.ones(3500)*1
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-1)
            yright_plot = yright_plot.reshape(-1,1)

            leftdy, leftc = 0,1
            rightdy, rightc = 0,-1

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]

    if yleft_plot[0]<0 or yright_plot[0]>0: 
        leftdy, leftc = 0,1.5
        rightdy, rightc = 0,-1.5
        yleft_plot = np.ones(3500)*1.5
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(3500)*(-1.5)
        yright_plot = yright_plot.reshape(-1,1)

        flag = False
        left_fit = [flag, leftdy,leftc]
        right_fit = [flag, rightdy,rightc]

    ################ Steering Angle #################

    if left_fit[0] == False:
        # theta = atan((left_fit[1]+right_fit[1])/2)
        theta = atan((leftdy+rightdy)/2)*180/pi
    else:
        if direction < 0 :
            # theta = math.acos(right_fit[0]/(right_fit[0]-yright_plot[0]))
            theta = acos(right_r/(right_r-yright_plot[0]))*180/pi
        else:
            # theta = math.acos(left_fit[0]/(left_fit[0]+yleft_plot[0]))
            theta = acos(left_r/(left_r+yleft_plot[0]))*180/pi

    if abs(theta) < 2:
        leftdy, leftc = 0,1.5
        rightdy, rightc = 0,-1.5
        yleft_plot = np.ones(3500)*1.5
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(3500)*(-1.5)
        yright_plot = yright_plot.reshape(-1,1)

        flag = False
        left_fit = [flag, leftdy,leftc]
        right_fit = [flag, rightdy,rightc]

    left_lane = np.append(xleft_plot,yleft_plot,axis =1)
    left_lane = left_lane[left_lane[:,1]<15]
    left_lane = left_lane[left_lane[:,1]>-15]
    right_lane = np.append(xright_plot,yright_plot,axis =1)
    right_lane = right_lane[right_lane[:,1]<15]
    right_lane = right_lane[right_lane[:,1]>-15]
    return left_lane, right_lane, left_fit, right_fit, theta



def line_equation(x,line_fit): 
    line_dy = line_fit[1]
    line_c = line_fit[2]
    y = line_dy*x+line_c
    return y


def curve_equation(x,curve_fit): 
    r = curve_fit[0]
    a = curve_fit[1][0]
    b = curve_fit[1][1]
    if b<0: y = (r**2-(x-a)**2)**0.5 +b
    else: y = -(r**2-(x-a)**2)**0.5 +b
    return y

def circle(y,a,b,r):
    return (r**2-(y-b)**2)**0.5 +a


def circle_plot(x,center,r):
    a = center[0]
    b = center[1]
    if b<0: y = (r**2-(x-a)**2)**0.5 +b
    else: y = -(r**2-(x-a)**2)**0.5 +b
    return y



def invadeROI(point, left_fit, right_fit):
    # if left_fit[0] == False:

    if left_fit[0] == False:
        y_left = line_equation(point[0], left_fit)
        y_right = line_equation(point[0], right_fit)

    else: 
        y_left = curve_equation(point[0], left_fit)
        y_right = curve_equation(point[0], right_fit)

    # if np.all(point[1]<y_left , point[1] > y_right , 5 < point[0] ,point[0] < 40): invade = True
    if point[1]<y_left and point[1] > y_right and 5 < point[0] and point[0] < 40: invade = True
    else: invade = False
    return invade


# def roi_box(left_lane, right_lane, line1_fit, line2_fit):
#     line1pred = line1_fit.predict(left_lane[:,0]).reshape([len1,1])
#     line2pred = line2_fit.predict(right_lane[:,0]).reshape([len2,1])

#     left_max = left_lane[:][np.argmax(line1pred),:2]
#     left_min = left_lane[:][np.argmin(line1pred),:2]

#     left_min = left_lane[:][np.argmin(line1pred),:2]


