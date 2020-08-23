import numpy as np
from math import sqrt, pi
import random

######################### Define Function ########################

def get_angle(input_list):
    angle = math.atan2(input_list[1], input_list[0])
    if input_list[1]<0:
        angle = angle+2*pi
    return angle*180/pi

def sortline_co(line):
    length = len(line[:][:,0])
    linedict = {}

    for i in range(0,length):
        linedict[line[:][i,0]] = line[:][i,0:2]

    linedict_sorted = sorted(linedict.items())
    line_sorted = np.empty([0,2])

    for j in range(0,length):
        line_sorted = np.append(line_sorted, [linedict_sorted[j][1]],axis = 0)

    return line_sorted ,length

# def sortline_angle(line):
#     length = len(line[:][:, 0])
#     tmp_x = sum(line[:][:, 0])/length
#     tmp_y = sum(line[:][:, 1]) / length
#     inner_point = [tmp_x, tmp_y]
#     linedict = {}
#     linevectors = line - inner_point

#     listangle = list(map(get_angle, linevectors))

#     for i in range(0, length):
#         # line1dict[xline1[i]] = [xline1[i],yline1[i]]
#         linedict[listangle[i]] = line[:][i, :]
#     linedict_sorted = sorted(linedict.items())

#     listangle = sorted(listangle)

#     line_sorted = np.empty([0, 2])

#     length = len(linedict_sorted)

#     for j in range(0, length):
#         line_sorted = np.append(line_sorted, [linedict_sorted[j][1]], axis=0)

#     for i in range(0, length - 1):
#         theta = abs(listangle[i] - listangle[i + 1])
#         if 180 < theta:
#             move = line_sorted[:i + 1]
#             line_sorted = line_sorted[i + 1:]
#             line_sorted = np.append(line_sorted, move, axis=0)

#     return line_sorted

########################################################################
############################ Main Function #############################
########################################################################


def calcurvature(lidar):

    curve_sorted, length = sortline_co(lidar)

    iternum = 4
    r_list = np.empty((1, 3))
    
    # for i in range(iternum):
    #     idx1 = random.randint(0, length//3)
    #     idx2 = random.randint(idx1+3, idx1+length//3)
    #     idx3 = idx2*2-idx1

    #     p1 = curve_sorted[idx1]
    #     p2 = curve_sorted[idx2]
    #     p3 = curve_sorted[idx3]
    #     print(idx1, ' ', idx2, ' ', idx3)
    #     print(length)

    #     a = sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    #     b = sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1])** 2)
    #     c = sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1])** 2)

    #     q = (a**2+b**2-c**2)/(2*a*b)
    #     r = c/(2*sqrt(1-q**2))

    #     r_list = np.append(r_list, r)

    #     if p2[1]>p1[1]: direction = -1
    #     else: direction = 1

    idx1 = 0
    idx2 = length//2
    idx3 = length-1

    p1 = curve_sorted[idx1]
    p2 = curve_sorted[idx2]
    p3 = curve_sorted[idx3]
    print(idx1, ' ', idx2, ' ', idx3)
    print(length)

    a = (p2[0]*p2[0]+p2[1]*p2[1])
    b = (p1[0]**2 + p1[1]**2-a)/2
    c = (a-p3[0]**2-p3[1]**2)/2
    d = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    print('d: ', d)

    cx = (b*(p2[1] - p3[1]) - c*(p1[1] - p2[1])) / d
    cy = ((p1[0] - p2[0]) * c - (p2[0] - p3[0]) * b) / d

    
    r = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    # q = (a**2+b**2-c**2)/(2*a*b)
    # r = c/(2*sqrt(1-q**2))

    r_list = np.append(r_list, r)

    if p2[1]>cy: direction = -1
    else: direction = 1

    r_avg = sum(r_list)/len(r_list)
    print(r_avg)
    
    center = np.array([cx,cy])
    print(center)
    return r_avg, center, direction






