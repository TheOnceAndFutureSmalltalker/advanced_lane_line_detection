import numpy as np
import cv2
import glob
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanelines as ll

# For some test images, run through entire sequence ending in the
# original image with lane lines drawn on it

# define initial values/constants
filenames = ['straight_lines1', 'straight_lines2', 'test1', 'test2', \
             'test3', 'test4', 'test5', 'test6']
i = 1
x_thresh = (20, 100)
s_thresh = (160, 255)
src = np.float32([[207,720], [595,450], [684,450], [1100,720]])
dst = np.float32([[337,720], [337,0], [970,0], [970,720]])

# instantiate and initialize processing objects
cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
gt = ll.GradientTransformer()
ct = ll.ColorTransformer() 
pw = ll.PerspectiveWarper(src, dst)
llf = ll.LaneLineFinder()

for filename in filenames:
    original = scipy.misc.imread('test_images/' + filename + '.jpg')
    undistorted = cc.undistort(original)
    absx_bin = gt.abs_thresh(undistorted, orient='x', thresh=x_thresh)
    s_bin = ct.s_thresh(undistorted, thresh=s_thresh)
    filtered_bin = np.zeros_like(absx_bin)
    filtered_bin[(absx_bin == 1) | (s_bin == 1)] = 1
    warped_bin = pw.warp(filtered_bin)
    llf.reset() # reset to do full search each time
    ploty, left_fitx, right_fitx = llf.find_lane_lines(warped_bin, False)
    warped_lane_mask = llf.lane_mask()
    unwarped_lane_mask = pw.unwarp(warped_lane_mask)
    final = cv2.addWeighted(original, 1, unwarped_lane_mask, 0.3, 0)
    scipy.misc.imsave('output_images/full_test_' + filename + '.jpg', final)
    i += 1






