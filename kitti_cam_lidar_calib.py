import numpy as np
import open3d as o3d
import time
import copy
import loadData
import cv2


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

path = "/home/jinwj1996/2011_09_26/2011_09_26_drive_0005_sync/"
path_lidar = path + "velodyne_points/data/"
#path_csv = path + "csvdata/"
path_image = path + "image_00/data/"

file_list = loadData.load_data(path_lidar)
image_list = loadData.load_data(path_image)

cv2.namedWindow('Show Image')

img = cv2.imread(path_image + image_list[0], cv2.IMREAD_COLOR)
files = file_list[0]

pointcloud = np.fromfile(path_lidar+files,dtype = np.float32)
pointcloud = pointcloud.reshape(-1,4)

# Cropping
pointcloud = pointcloud[(pointcloud[:,2] >= -1.40)]
pointcloud = pointcloud[(pointcloud[:,0] >= 0)]
pointcloud = pointcloud[(pointcloud[:,0] <= 30)]
pointcloud = pointcloud[(pointcloud[:,1] >= -10)]
pointcloud = pointcloud[(pointcloud[:,1] <= 10)]

pointcloud[:,3] = 1.0


Y = (Calibration_matrix @ pointcloud.T).T


Point_x = Y[:,0] / Y[:,2]
Point_y = Y[:,1] / Y[:,2]


print(Point_x)
print(Point_y)

cv2.circle(img, (Point_x, Point_y), 5, cv::Scalar(0, 0, 1), -1)

cv2.imshow("Show Image", img)
cv2.waitKey(1)

#for files in file_list:
#    dt = 0.2
