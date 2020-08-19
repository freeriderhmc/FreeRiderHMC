# import numpy as np

# lidarpoint3D = np.array([[5,5,5], [5.5,5.5,5.5], [6,6,6], [7,7,7]])

# # (x,y)
# lidarpoint2D = np.array([[1,0], [2,0], [1,1], [2,2]])

# # 3 X 3
# semanticMap = np.array([[0,1,1],
#                         [0,1,0],
#                         [0,0,2]])

# # label == 1 index -> [[x,y], ~]
# #print( np.array([np.where(semanticMap == 1)[0],  np.where(semanticMap == 1)[1]] ).T )

# classification = np.array(list(map(lambda x : semanticMap[x[1], x[0]], lidarpoint2D)))

# print(classification)

# # index = np.array([np.where(semanticMap == 1)[0],  np.where(semanticMap == 1)[1]] ).T
# # print(lidarpoint2D)
# # print(index)
# #print(np.where(semanticMap == 1)[0][:])
# #print(np.where(semanticMap == 1)[1])

# #print(np.where(lidarpoint2D[:] == index[:]))

# #print(lidarpoint2D[index[:,0], index[:,1] ])


import numpy as np
import open3d as o3d
import time
import copy
import loadData
import cv2
from math import atan, pi
from matplotlib import pyplot as plt
from curve import curve, invadeROI




RT = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
               [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
               [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
               [0.0, 0.0, 0.0, 1.0]])

R_rect_00 = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0],
                      [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0],
                      [7.402527e-03, 4.351614e-03, 9.999631e-01, 0.0],
                      [0.0, 0.0, 0.0, 1]])

P_rect_00 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                      [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                      [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

Calibration_matrix = P_rect_00 @ R_rect_00 @ RT


path = "./"
path_lidar = path + "lidar/"
#path_csv = path + "csvdata/"
path_image = path + "image/"
path_semanticMap = path + "semanticMap/"

file_list = loadData.load_data(path_lidar)
image_list = loadData.load_data(path_image)
map_list = loadData.load_data(path_semanticMap)

#cv2.namedWindow('Show Image')
i = 1
img = cv2.imread(path_image + image_list[i], cv2.IMREAD_COLOR)
print(img.shape)
semanticMap = np.fromfile(path_semanticMap + map_list[i], dtype = np.int64)
semanticMap = semanticMap.reshape(img.shape[0], img.shape[1])
# semanticMap = semanticMap.reshape(375, 1242)

# files = file_list[0]
# semanticMap = map_list[0]
#print(files)
pointcloud = np.fromfile(path_lidar+file_list[i],dtype = np.float32)
pointcloud = pointcloud.reshape(-1,4)

# Cropping
# pointcloud = pointcloud[(pointcloud[:,2] >= -1.40)]
# pointcloud = pointcloud[(pointcloud[:,2] >= -1.5)]
# pointcloud = pointcloud[(pointcloud[:,2] <= 2.0)]
pointcloud = pointcloud[(pointcloud[:,1] >= -10)]
pointcloud = pointcloud[(pointcloud[:,1] <= 10)]
pointcloud[:,3] = 1.0

#pc_front = pointcloud

pc_back = pointcloud[(pointcloud[:,0] >= -10)]
pc_back = pc_back[(pc_back[:,0] <= 0)]
pc_front = pointcloud[(pointcloud[:,0] >= 0)]
pc_front = pc_front[(pc_front[:,0] <= 40)]

pointcloud = np.append(pc_back, pc_front, axis = 0)

label_back = np.zeros(len(pc_back))
print(pc_back.shape, pc_front.shape)



Y = (Calibration_matrix @ pc_front.T).T
#Y = np.dot(Calibration_matrix, pc_front.T).T

Y[:,0] = Y[:,0] / Y[:,2]
Y[:,1] = Y[:,1] / Y[:,2]

Y = Y[:, :2]
# Y = Y[(0 <= Y[:,0])]
# Y = Y[(Y[:,0] < 1241)]
# Y = Y[(0 <= Y[:,1])]
# Y = Y[(Y[:,1] < 376)]

Y = np.int_(Y)
# print(Y.shape)
# Point_x = Y[:,0] / Y[:,2]
# Point_y = Y[:,1] / Y[:,2]

# Point_x = Point_x.astype('int')
# Point_y = Point_y.astype('int')

# print(Point_x)
# print(Point_y)
start = time.time()
label_front = np.array(list(map(lambda x : semanticMap[x[1], x[0]] if 0 <= x[0] < img.shape[1] and 0 <= x[1] < img.shape[0] else 0, Y)))

pointcloud = pointcloud[:,:3]

pc_label = np.append(label_back, label_front, axis = 0)
#print(label_back.shape, label_front.shape, pc_label.shape)

# Get Lane Pointcloud : 24
label_24_index = np.where(pc_label == 24)[0]
pc_label_24 = pointcloud[label_24_index]

# Get Car Pointcloud : 55
label_55_index = np.where(pc_label == 55)[0]
pc_label_55 = pointcloud[label_55_index]

# Get human Pointcloud : ??
# label_??_index = np.where(pc_label == ??)[0]
# pointcloud_label_?? = pointcloud[label_??_index]

# pointcloud_else = np.delete(pointcloud, label_24_index, 0)
# pointcloud_else = np.delete(pointcloud_else, label_55_index, 0)
print("total : ", pointcloud.shape)
print("lane : ", pc_label_24.shape)
print("car : ", pc_label_55.shape)
#print("else: ", pointcloud_else.shape)
# pointcloud_else = np.delete(pointcloud, label_??_index, 0)
print("Mapping time : ", time.time() - start)

############################### Calculate Curve ##############################
start = time.time()
left_lane, right_lane, left_fit ,right_fit = curve(pc_label_24)

leftdy = left_fit.coef_
rightdy = right_fit.coef_
leftc = left_fit.intercept_
rightc = right_fit.intercept_


# heading_ang = atan2(1,(leftdy+rightdy/2))*180/pi
heading_ang = atan(leftdy+rightdy/2)*180/pi
print('degree:',heading_ang)
print("Curve time : ", time.time() - start)

print(invadeROI([1.5,6],leftdy, rightdy,leftc, rightc))


################################## Plot ###################################

left_pred = left_fit.predict(left_lane[:,0].reshape(-1,1)).reshape(-1,1)
right_pred = right_fit.predict(right_lane[:,0].reshape(-1,1)).reshape(-1,1)


plt.figure()
plt.plot(pointcloud[:,0], pointcloud[:,1], 'bo', markersize = 0.8)
plt.plot(pc_label_24[:,0], pc_label_24[:,1], 'ro', markersize = 0.8)
plt.plot(pc_label_55[:,0], pc_label_55[:,1], 'go', markersize = 0.8)
plt.plot(left_lane[:][:,0], left_pred)
plt.plot(right_lane[:][:,0],right_pred)
plt.plot(6,1.5,'mo',markersize =4)
plt.xlim(-10,30)
plt.ylim(-20,20)
plt.show()

# print(pointcloud.shape)
# print(pointcloud_label_30.shape)
# print(pointcloud_else.shape)

#lambda x : (for i in Y : cv2.circle(img, (x[0], x[1]), 3, (0,0,255), -1))

# lane : 24, car : 55, human : unknown
start = time.time()
for i in range(0, len(Y)):
    #cv2.circle(img, (Y[i,0], Y[i,1]), 1, (0, 0, 255), -1)
    if label_front[i] == 24:
        cv2.circle(img, (Y[i,0], Y[i,1]), 1, (0, 0, 255), -1)
    else:
        cv2.circle(img, (Y[i,0], Y[i,1]), 1, (0, 0, 1), -1)

print("circle time : ", time.time() - start)
cv2.imshow("Show Image", img)
cv2.waitKey(0)

