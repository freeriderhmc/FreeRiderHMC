import numpy as np
import math
from sklearn.linear_model import LinearRegression

# prev_leftdy = 0
# prev_rightdy = 0
# prev_leftc = 0
# prev_rightc = 0

def curve(pc_lane):

    left_lane = pc_lane[pc_lane[:][:,1]>1]
    left_lane = left_lane[left_lane[:][:,1]<3]
    left_lane = left_lane[left_lane[:][:,0]<40]
    right_lane = pc_lane[pc_lane[:][:,1]<-1] 
    right_lane = right_lane[right_lane[:][:,1]>-3]
    right_lane = right_lane[right_lane[:][:,0]<40]

    stop_lane = pc_lane[pc_lane[:][:,1]<1]
    stop_lane = stop_lane[stop_lane[:][:,1]>-1]
    stop_lane = stop_lane[stop_lane[:][:,0]<10]
    

    if len(stop_lane)>20: stop = True
    else: stop = False
    print('stop: ',stop)
    
    line_fitter1 = LinearRegression()
    line_fitter2 = LinearRegression()

    xleft_plot = np.arange(5,40,0.01).reshape(-1,1)
    xright_plot = np.arange(5,40,0.01).reshape(-1,1)

    # left_lane = pc_lane[pc_lane[:][:,1]<-2]
    # left_lane = left_lane[left_lane[:][:,1]>-4]
    # left_lane = left_lane[left_lane[:][:,0]<15]
    # right_lane = pc_lane[pc_lane[:][:,1]>1] 
    # right_lane = right_lane[right_lane[:][:,1]<3]
    # right_lane = right_lane[right_lane[:][:,0]<15]

    if len(left_lane) <= 10 and len(right_lane) <= 10:
        leftdy, leftc = 0,1.5
        rightdy, rightc = 0,-1.5
        yleft_plot = np.ones(3500)*1.5
        yleft_plot = yleft_plot.reshape(-1,1)
        yright_plot = np.ones(3500)*(-1.5)
        yright_plot = yright_plot.reshape(-1,1)

    elif len(left_lane)<10:
        xright = right_lane[:][:,0].reshape(-1,1)
        yright = right_lane[:][:,1].reshape(-1,1)
        right_fit = line_fitter2.fit(xright,yright)

        rightdy = right_fit.coef_
        rightc = right_fit.intercept_

        leftdy, leftc = rightdy,rightc+3
        if -0.07 < rightdy < 0.07:
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
            yleft_plot = yright_plot+3            
        else:
            yleft_plot = np.ones(3500)*1.5
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-1.5)
            yright_plot = yright_plot.reshape(-1,1)
           

    elif len(right_lane)<10:
        xleft = left_lane[:][:,0].reshape(-1,1)
        yleft = left_lane[:][:,1].reshape(-1,1)
        left_fit = line_fitter1.fit(xleft,yleft)

        leftdy = left_fit.coef_
        leftc = left_fit.intercept_

        rightdy, rightc = leftdy,leftc-3

        if -0.07 < leftdy < 0.07:
            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = yleft_plot-3
        else:
            yleft_plot = np.ones(3500)*1.5
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-1.5)
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
        leftc = left_fit.intercept_
        rightc = right_fit.intercept_

        if -0.07 < leftdy < 0.07 and -0.07 < rightdy < 0.07 :
            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
        elif -0.07 < leftdy < 0.07:
            yleft_plot = left_fit.predict(xleft_plot).reshape(-1,1)
            yright_plot = yleft_plot-3            
        elif -0.07 < rightdy < 0.07:
            yright_plot = right_fit.predict(xright_plot).reshape(-1,1)
            yleft_plot = yright_plot+3            
        else:
            yleft_plot = np.ones(3500)*(1.5)
            yleft_plot = yleft_plot.reshape(-1,1)
            yright_plot = np.ones(3500)*(-1.5)
            yright_plot = yright_plot.reshape(-1,1)

    yleft_plot = yleft_plot-0.5
    yright_plot = yright_plot+0.5
    left_lane = np.append(xleft_plot,yleft_plot,axis =1)
    right_lane = np.append(xright_plot,yright_plot,axis =1)


    
    return left_lane, right_lane, [leftdy,leftc], [rightdy, rightc], stop



def line_equation(x,line_dy,line_c): 
    y = (x*line_dy+line_c)
    return y


def invadeROI(point, left_fit, right_fit):
    leftdy = left_fit[0]
    leftc = left_fit[1]-0.5
    rightdy = right_fit[0]
    rightc = right_fit[1]+0.5
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


