import numpy as np
import cv2
from matplotlib import pyplot as plt

################################################################################
######################### Calibration Matrix ###################################

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

###############################################################################

def addImage(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    return img1, img2


img1, img2 = addImage('./48/0000000048.png', './48/semantic0000000048.png')
pointcloud = np.fromfile('./48/lidar0000000048.bin', dtype = np.float32)
semanticMap = np.fromfile('./48/semap0000000048.bin', dtype = np.int64)


pointcloud = pointcloud.reshape(-1,4)

pointcloud_plot = pointcloud[:,:3]
pointcloud_plot = pointcloud_plot[(pointcloud_plot[:,0] >= -10)]
pointcloud_plot = pointcloud_plot[(pointcloud_plot[:,0] <= 60)]
pointcloud_plot = pointcloud_plot[(pointcloud_plot[:,1] >= -20)]
pointcloud_plot = pointcloud_plot[(pointcloud_plot[:,1] <= 20)]
pointcloud_plot = (np.array([[0,-1,0], [1,0,0], [0,0,1]]) @ pointcloud_plot.T).T 

plt.plot(pointcloud_plot[:,0], pointcloud_plot[:,1], 'ko', markersize = 0.6)
plt.xlim(-30,30)
plt.ylim(-10,60)
plt.show()

# sensor fusion
pointcloud = pointcloud[(pointcloud[:,0] >= 0)]
Y = (Calibration_matrix @ pointcloud.T).T
Y[:,0] = Y[:,0] / Y[:,2]
Y[:,1] = Y[:,1] / Y[:,2]
Y = Y[:, :2]
Y = np.int_(Y)
label = np.array(list(map(lambda x : semanticMap[x[1], x[0]] if 0 <= x[0] < img1.shape[1] and 0 <= x[1] < img1.shape[0] else 0, Y)))

# Show image
cv2.imshow('image', img1)
k = cv2.waitKey(0)
if k == 27: # esc key
    cv2.destroyAllWindows()

cv2.imshow('semantic image', img2)
k = cv2.waitKey(0)
if k == 27: # esc key
    cv2.destroyAllWindows()

add_img = cv2.add(img1, img2)
cv2.imshow('add', add_img)
k = cv2.waitKey(0)
if k == 27: # esc key
    cv2.destroyAllWindows()


# Fusion 3D Image
