import numpy as np
import math
# from sklearn.linear_model import LinearRegression
from curvature import calcurvature
from scipy.optimize import curve_fit

# prev_leftdy = 0
# prev_rightdy = 0
# prev_leftc = 0
# prev_rightc = 0

def curve(left_lane, right_lane):

    xleft_plot = np.arange(5,40,0.01).reshape(-1,1)
    xright_plot = np.arange(5,40,0.01).reshape(-1,1)

    if len(left_lane) < 10 and len(right_lane) < 10:
        leftdy, leftc = 0,2
        rightdy, rightc = 0,-2
        yleft_plot = np.ones(3500)*2
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(3500)*(-2)
        yright_plot = yright_plot.reshape(-1,1)

        flag = False
        left_fit = [flag, leftdy,leftc]
        right_fit = [flag, rightdy,rightc]

    elif len(left_lane)< 10:
        left_r, left_center, direction = calcurvature(left_lane)       
        # left_center = direction*left_r +1.5         
        right_r = left_r+direction*3
        right_center = left_center

        if 20 <right_r:
            yright_plot = curve_equation(xright_plot, right_fit).reshape(-1,1)
            yleft_plot = yright_plot+3                        

            left_fit = [left_r,left_center, left_direction]
            right_fit = [right_r,right_center, right_direction]
        else:
            yleft_plot = np.ones(3500)*2
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-2)
            yright_plot = yright_plot.reshape(-1,1)

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]
           

    elif len(right_lane)<10:
        left_r, left_center, direction = calcurvature(left_lane)       
        # left_center = direction*left_r +1.5         
        right_r = left_r+direction*3
        # right_center = direction*right_r -1.5     
        right_center = left_center
        

        if 20 <left_r:
            yleft_plot = curve_equation(xleft_plot, left_fit).reshape(-1,1)
            yright_plot = yleft_plot+3           

            left_fit = [left_r,left_center, left_direction]
            right_fit = [right_r,right_center, right_direction]
        else:
            yleft_plot = np.ones(3500)*2
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-2)
            yright_plot = yright_plot.reshape(-1,1)

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]

    else:
        left_r, left_center, left_direction = calcurvature(left_lane)       
        # left_center = left_direction*left_r +1.5 
        right_r, right_center, right_direction = calcurvature(right_lane)       
        # right_center = right_direction*right_r -1.5

        left_fit = [left_r,left_center, left_direction]
        right_fit = [right_r,right_center, right_direction]
        
        if 20 <left_r and 20 <right_r:
            yleft_plot = curve_equation(xleft_plot, left_fit).reshape(-1,1)
            yright_plot = curve_equation(xright_plot, right_fit).reshape(-1,1)
            
            left_fit = [left_r,left_center, left_direction]
            right_fit = [right_r,right_center, right_direction]

        elif 20 <left_r:
            yleft_plot = curve_equation(xleft_plot, left_fit).reshape(-1,1)
            yright_plot = yleft_plot+3       

            left_fit = [left_r,left_center, left_direction]
            right_fit = [right_r,right_center, right_direction]

        elif 20 <right_r:
            yright_plot = curve_equation(xright_plot, right_fit).reshape(-1,1)
            yleft_plot = yright_plot+3                       

            left_fit = [left_r,left_center, left_direction]
            right_fit = [right_r,right_center, right_direction]           
        else:
            yleft_plot = np.ones(3500)*2
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-2)
            yright_plot = yright_plot.reshape(-1,1)

            flag = False
            left_fit = [flag, leftdy,leftc]
            right_fit = [flag, rightdy,rightc]

    # print(yleft_plot)
    left_lane = np.append(xleft_plot,yleft_plot,axis =1)
    right_lane = np.append(xright_plot,yright_plot,axis =1)
    # print(right_lane)
    print('r: ',right_r)
    print(left_r)
    return left_lane, right_lane, left_fit, right_fit



def line_equation(x,line_fit): 
    line_dy = line_fit[1]
    line_c = line_fit[2]
    y = (x-line_c)/line_dy
    return y


def curve_equation(x,curve_fit): 
    x = np.array(x)
    r = curve_fit[0]
    center = curve_fit[1]
    direction = curve_fit[2]
    y = center-direction*(r**2-x**2)**0.5
    return y

def circle(y,a,b,r):
    return (r**2-(y-b)**2)**0.5 +a

def invadeROI(point, left_fit, right_fit):
    if left_fit[0] == False:
        y_left = line_equation(point, left_fit)
        y_right = line_equation(point, right_fit)

    else: 
        left_r = left_fit[1]
        left_center = left_fit[2]
        left_direction = left_fit[3]
        right_r= right_fit[1]
        right_center = right_fit[2]
        right_direction = right_fit[3]
        y_left = curve_equation(point, left_fit)
        y_right = curve_equation(point, right_fit)

    if point[1]>y_left and point[1]<y_right: invade = True
    else: invade = False
    return invade


# def roi_box(left_lane, right_lane, line1_fit, line2_fit):
#     line1pred = line1_fit.predict(left_lane[:,0]).reshape([len1,1])
#     line2pred = line2_fit.predict(right_lane[:,0]).reshape([len2,1])

#     left_max = left_lane[:][np.argmax(line1pred),:2]
#     left_min = left_lane[:][np.argmin(line1pred),:2]

#     left_min = left_lane[:][np.argmin(line1pred),:2]


