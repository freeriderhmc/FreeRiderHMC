import numpy as np
import math
from sklearn.linear_model import LinearRegression


def curve(pc_lane):
    left_lane = pc_lane[pc_lane[:][:,1]<-1]
    left_lane = left_lane[left_lane[:][:,1]>-3]
    left_lane = left_lane[left_lane[:][:,0]<20]
    right_lane = pc_lane[pc_lane[:][:,1]>1] 
    right_lane = right_lane[right_lane[:][:,1]<3]
    right_lane = right_lane[right_lane[:][:,0]<20]

    # left_lane = pc_lane[pc_lane[:][:,1]<-2]
    # left_lane = left_lane[left_lane[:][:,1]>-4]
    # left_lane = left_lane[left_lane[:][:,0]<15]
    # right_lane = pc_lane[pc_lane[:][:,1]>1] 
    # right_lane = right_lane[right_lane[:][:,1]<3]
    # right_lane = right_lane[right_lane[:][:,0]<15]

    line_fitter1 = LinearRegression()
    line_fitter2 = LinearRegression()

    len1 = len(left_lane[:][:,0])
    len2 = len(right_lane[:][:,0])

    xline1 = left_lane[:][:,0].reshape(-1,1)
    yline1 = left_lane[:][:,1].reshape(-1,1)
    xline2 = right_lane[:][:,0].reshape(-1,1)
    yline2 = right_lane[:][:,1].reshape(-1,1)

    line1_fit = line_fitter1.fit(xline1,yline1)
    line2_fit = line_fitter2.fit(xline2,yline2)
    line1dy = line1_fit.coef_
    line2dy = line2_fit.coef_

    # line1c = line1_fit.intercept_
    # line2c = line2_fit.intercept_
    # line1pred = line1_fit.predict(xline1).reshape([len1,1])
    # line2pred = line2_fit.predict(xline2).reshape([len2,1])

    return left_lane, right_lane, line1_fit, line2_fit

def line_equation(x,line_dy,line_c): 
    y = (x-line_c)/line_dy
    return y

def invadeROI(point, leftdy, rightdy, leftc, rightc):
    y_left = line_equation(point[0], leftdy, leftc)
    y_right = line_equation(point[0], rightdy, rightc)
    
    if point[1]>y_left and point[1]<y_right: invade = True
    else: invade = False
    return invade


def roi_box(left_lane, right_lane, line1_fit, line2_fit):
    line1pred = line1_fit.predict(left_lane[:,0]).reshape([len1,1])
    line2pred = line2_fit.predict(right_lane[:,0]).reshape([len2,1])

    left_max = left_lane[:][np.argmax(line1pred),:2]
    left_min = left_lane[:][np.argmin(line1pred),:2]

    left_min = left_lane[:][np.argmin(line1pred),:2]


