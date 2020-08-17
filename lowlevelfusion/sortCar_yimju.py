import numpy as np
import open3d as o3d
import math
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# import sortline as sl

############################## Macro ###############################
pi = 3.141592653589793238

######################### Define Function ##########################
def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    # if angle<-pi/4:
    #     angle = angle + pi
    
    return angle



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

if __name__ == "__main__":
    print("Error.. Why sortCar Module execute")