import numpy as np
import math
from sklearn.linear_model import LinearRegression

# prev_leftdy = 0
# prev_rightdy = 0
# prev_leftc = 0
# prev_rightc = 0

def curve2line(pc_lane, th, y_start,step):
    step = int(35/step)
    x_limit = th*step
    # print(x_limit)
    ly_start = y_start[0]
    ry_start = y_start[1]

    left_lane = pc_lane[pc_lane[:][:,1]>ly_start-1]
    left_lane = left_lane[left_lane[:][:,1]<ly_start+1]
    left_lane = left_lane[left_lane[:][:,0]<x_limit]
    right_lane = pc_lane[pc_lane[:][:,1]<ry_start+1] 
    right_lane = right_lane[right_lane[:][:,1]>ry_start-1]
    right_lane = right_lane[right_lane[:][:,0]<x_limit]

    stop_lane = pc_lane[pc_lane[:][:,1]<1]
    stop_lane = stop_lane[stop_lane[:][:,1]>-1]
    stop_lane = stop_lane[stop_lane[:][:,0]<5]    

    if len(stop_lane)>20: stop = True
    else: stop = False
    # print('stop: ',stop)
    
    line_fitter1 = LinearRegression()
    line_fitter2 = LinearRegression()

    xleft_plot = np.arange(x_limit-step,x_limit,0.01).reshape(-1,1)
    # print(xleft_plot)
    xright_plot = np.arange(x_limit-step,x_limit,0.01).reshape(-1,1)

    # left_lane = pc_lane[pc_lane[:][:,1]<-2]
    # left_lane = left_lane[left_lane[:][:,1]>-4]
    # left_lane = left_lane[left_lane[:][:,0]<15]
    # right_lane = pc_lane[pc_lane[:][:,1]>1] 
    # right_lane = right_lane[right_lane[:][:,1]<3]
    # right_lane = right_lane[right_lane[:][:,0]<15]

    if len(left_lane) < 5 and len(right_lane) < 5:
        leftdy, leftc = 0,ly_start
        rightdy, rightc = 0,ry_start
        yleft_plot = np.ones(step*100)*ly_start
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(step*100)*ry_start
        yright_plot = yright_plot.reshape(-1,1)

    elif len(left_lane)<5:
        xright = right_lane[:][:,0].reshape(-1,1)
        yright = right_lane[:][:,1].reshape(-1,1)
        right_fit = line_fitter2.fit(xright,yright)
        rightdy = right_fit.coef_
        if -0.07 < rightdy < 0.07:
            rightc = right_fit.intercept_
            leftdy, leftc = rightdy,rightc+3
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
            yleft_plot = yright_plot+3            
        else:
            leftdy, leftc = 0,ly_start
            rightdy, rightc = 0,ry_start
            yleft_plot = np.ones(step*100)*ly_start
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(step*100)*ry_start
            yright_plot = yright_plot.reshape(-1,1)
           

    elif len(right_lane)<5:
        xleft = left_lane[:][:,0].reshape(-1,1)
        yleft = left_lane[:][:,1].reshape(-1,1)
        left_fit = line_fitter1.fit(xleft,yleft)
        leftdy = left_fit.coef_

        if -0.07 < leftdy < 0.07:            
            leftc = left_fit.intercept_
            rightdy, rightc = leftdy,leftc-3
            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = yleft_plot-3
        else:
            leftdy, leftc = 0,ly_start
            rightdy, rightc = 0,ry_start
            yleft_plot = np.ones(step*100)*ly_start
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(step*100)*ry_start
            yright_plot = yright_plot.reshape(-1,1)

    else:
        xleft = left_lane[:][:,0].reshape(-1,1)
        yleft = left_lane[:][:,1].reshape(-1,1)
        xright = right_lane[:][:,0].reshape(-1,1)
        yright = right_lane[:][:,1].reshape(-1,1)

        left_fit = line_fitter1.fit(xleft,yleft)
        right_fit = line_fitter2.fit(xright,yright)
        leftdy = left_fit.coef_
        rightdy = right_fit.coef_

        if -0.07 < leftdy < 0.07 and -0.07 < rightdy < 0.07 :            
            leftc = left_fit.intercept_
            rightc = right_fit.intercept_

            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
        elif -0.07 < leftdy < 0.07:
            leftc = left_fit.intercept_
            rightdy, rightc = leftdy,leftc-3

            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = yleft_plot-3       

        elif -0.07 < rightdy < 0.07:
            rightc = right_fit.intercept_
            leftdy, leftc = rightdy,rightc+3
            
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
            yleft_plot = yright_plot+3            
        else:
            leftdy, leftc = 0,ly_start
            rightdy, rightc = 0,ry_start
            yleft_plot = np.ones(step*100)*ly_start
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(step*100)*ry_start
            yright_plot = yright_plot.reshape(-1,1)

    if not isinstance(leftdy, int):
        leftdy = leftdy[0][0]
    if not isinstance(rightdy, int):
        rightdy = rightdy[0][0]
    if not isinstance(leftc, float):
        leftc = leftc[0]
    if not isinstance(rightc, float):
        rightc = rightc[0]

    # print(leftdy)
    # print(xright_plot)
    # print(yright_plot)
    left_lane = np.append(xleft_plot,yleft_plot,axis =1)
    right_lane = np.append(xright_plot,yright_plot,axis =1)
    left_fit = np.append([leftdy], [leftc],axis =0)
    right_fit = np.append([rightdy], [rightc],axis =0)
    return left_lane, right_lane, left_fit, right_fit, stop


def line_equation(x,line_dy,line_c): 
    y = (x*line_dy+line_c)
    return y


def invadeROI(point, left_fit, right_fit):
    # if point[0]<10: idx = 0
    # elif point[0]<20: idx = 1
    # elif point[0]<30: idx = 2
    # else: idx = 3

    # if point[0]<5: idx = 0
    # elif point[0]<10: idx = 1
    # elif point[0]<15: idx = 2
    # elif point[0]<20: idx = 3
    # elif point[0]<25: idx = 4
    # elif point[0]<30: idx = 5
    # elif point[0]<35: idx = 6
    # elif point[0]<40: idx = 7
    # else: idx = 8
    if point[0]<6: idx = 1
    elif point[0]<7: idx = 1
    elif point[0]<8: idx = 2
    elif point[0]<9: idx = 3
    elif point[0]<10: idx = 4
    elif point[0]<11: idx = 5
    elif point[0]<12: idx = 6
    elif point[0]<13: idx = 7
    elif point[0]<14: idx = 8
    elif point[0]<15: idx = 9
    elif point[0]<16: idx = 10
    elif point[0]<17: idx = 11
    elif point[0]<18: idx = 12

    # print(left_fit)
    leftdy = left_fit[idx][0]
    leftc = left_fit[idx][1]
    rightdy = right_fit[idx][0]
    rightc = right_fit[idx][1]
    y_left = line_equation(point[0], leftdy, leftc)
    y_right = line_equation(point[0], rightdy, rightc)

    
    if point[1]<y_left and point[1]>y_right: invade = True
    else: invade = False
    return invade


# def roi_box(left_lane, right_lane, line1_fit, line2_fit):
#     line1pred = line1_fit.predict(left_lane[:,0]).reshape([len1,1])
#     line2pred = line2_fit.predict(right_lane[:,0]).reshape([len2,1])

#     left_max = left_lane[:][np.argmax(line1pred),:2]
#     left_min = left_lane[:][np.argmin(line1pred),:2]

#     left_min = left_lane[:][np.argmin(line1pred),:2]


def curve(pc_lane,step):
    step = step
    y_start = [2.0,-2.0]
    # left_lane_list = np.empty([4,0,2])
    # right_lane_list = np.empty([4,0,2])
    # left_fit_list = np.empty([4,0,2])
    # right_fit_list = np.empty([4,0,2])
    left_lane, right_lane, left_fit, right_fit, stop = curve2line(pc_lane,1,y_start,step)
    left_lane_list = [left_lane]
    
    right_lane_list = [right_lane]
    # print(right_lane_list)
    left_fit_list = [left_fit]
    right_fit_list = [right_fit]
    stop_list = stop
    # print(left_lane[len(left_lane)-1])
    y_start =[left_lane[len(left_lane)-1][1], right_lane[len(right_lane)-1][1]]
    # print(y_start)

    for i in range(2,step):
        left_lane, right_lane, left_fit, right_fit, stop = curve2line(pc_lane,i,y_start,step)
        # print(left_lane)
        y_start =[left_lane[len(left_lane)-1][1], right_lane[len(right_lane)-1][1]]
        left_lane_list = np.append(left_lane_list,[left_lane], axis=0)
        # print(left_lane_list)
        right_lane_list = np.append(right_lane_list,[right_lane], axis=0)
        left_fit_list = np.append(left_fit_list,[left_fit], axis=0)
        right_fit_list = np.append(right_fit_list,[right_fit], axis=0)
    return left_lane_list, right_lane_list, left_fit_list, right_fit_list, np.array(stop_list)

