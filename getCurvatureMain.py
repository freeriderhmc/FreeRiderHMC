import matplotlib.pyplot as plt
import cv2 
import numpy as np
import pickle
from scipy.misc.pilutil import imread, imresize

from birdseye import BirdsEye
from lanefilter import LaneFilter
from curves import Curves
from helpers import roi

calibration_data = pickle.load(open("calibration_data.p", "rb" ))

matrix = calibration_data['camera_matrix']
distortion_coef = calibration_data['distortion_coefficient']

source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
destination_points = [(320, 0), (320, 720), (960, 720), (960, 0)]

p = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }

birdsEye = BirdsEye(source_points, destination_points, matrix, distortion_coef)
laneFilter = LaneFilter(p)
curves = Curves(number_of_windows = 9, margin = 100, minimum_pixels = 50, 
                ym_per_pix = 30 / 720 , xm_per_pix = 3.7 / 700)
                
def debug_pipeline(img):
    
  ground_img = birdsEye.undistort(img)
  binary_img = laneFilter.apply(ground_img)
  
  wb = np.logical_and(birdsEye.sky_view(binary_img), roi(binary_img)).astype(np.uint8)
  result = curves.fit(wb)
    
  left_curve =  result['pixel_left_best_fit_curve']
  right_curve =  result['pixel_right_best_fit_curve']
    
  left_radius =  result['left_radius']
  right_radius =  result['right_radius']
  
  projected_img = birdsEye.project(ground_img, binary_img, left_curve, right_curve)
    
  return projected_img, left_radius, right_radius
  
def verbose_pipeline(img):
  pro_img, lr, rr = debug_pipeline(img)


  offset = [0, 320, 640, 960]
  width, height = 320,180

  text_l = "left r: " + str(np.round(lr, 2)) 
  text_r = " right r: " + str(np.round(rr, 2))
    
  cv2.putText(pro_img, text_l, (20, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  cv2.putText(pro_img, text_r, (250, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

  return pro_img
  
path = "test_images/test1.jpg" ##
img = imread(path)
plt.imshow(verbose_pipeline(img))
