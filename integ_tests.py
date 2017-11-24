import numpy as np
import glob
import scipy.misc
import lanelines as ll

# image notes
# cv2.imread() BGR
# mpimg.imread() RGB
# scipy.misc.imread() RGB
# plt.imshow(img) or plt.imshow(img, cmap='gray')



# CamerCalibrator
# use CameraCalibration to read a checkerboard image 
# save it as cc_initial_checkerboard_image.jpg
# undistort it and save it as cc_undistored_checkerboard_image.jpg
cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
img = scipy.misc.imread('camera_cal/calibration2.jpg')
scipy.misc.imsave('output_images/cc_initial_checkerboard_image.jpg', img)
undistorted = cc.undistort(img)
scipy.misc.imsave('output_images/cc_undistorted_checkerboard_image.jpg', \
                  undistorted)
# use CameraCalibration to read a road image 
# save it as cc_initial_road_image.jpg
# undistort it and save it as cc_undistored_road_image.jpg
img = scipy.misc.imread('test_images/signs_vehicles_xygrad.png')
scipy.misc.imsave('output_images/cc_initial_road_image.jpg', img)
undistorted = cc.undistort(img)
scipy.misc.imsave('output_images/cc_undistorted_road_image.jpg', \
                  undistorted)



# PerspectiveTransformer
# read straight_lines1.jpg and save as pt_straight_lines_distorted.jpg
img = scipy.misc.imread('test_images/straight_lines1.jpg')
scipy.misc.imsave('output_images/pt_straight_lines_distorted.jpg', img)
# use CameraCalibrator to undistort it and save as
# pt_straight_lines_undistorted.jpg
undistorted = cc.undistort(img)
scipy.misc.imsave('output_images/pt_straight_lines_undistorted.jpg', \
                  undistorted)
# view this image to determine src and dst transform coordinates
src = np.float32([[207,720], [595,450], [684,450], [1100,720]])
dst = np.float32([[337,720], [337,0], [970,0], [970,720]])
# use PerspectiveTransformer to transform the image to a birds-eye view
# and save as pt_straight_lines_warped.jpg  
pw = ll.PerspectiveWarper(src, dst)
warped = pw.warp(undistorted)
scipy.misc.imsave('output_images/pt_straight_lines_warped.jpg', warped)
# use PerspectiveTransformer to transform the image back
# and save as pt_straight_lines_unwarped.jpg
unwarped = pw.unwarp(warped)
scipy.misc.imsave('output_images/pt_straight_lines_unwarped.jpg', unwarped)



# GradientFilter
# read test1.jpg thru test6.jpg
# for each save as test*_initial.jpg
# use GradientFilter to perform abs, mag, and dir gradients
# save each version as test*_gf_type_thresholdvalues.jpg
filenames = glob.glob('test_images/test*.jpg')
i = 1
x_thresh = (20,100)
gt = ll.GradientTransformer()
for filename in filenames:
    img = scipy.misc.imread(filename)
    outname = 'output_images/test{0}_initial.jpg'.format(i)
    scipy.misc.imsave(outname, img)
    abs_bin = gt.abs_thresh(img, orient='x', thresh=(20, 100))
    abs_bin[abs_bin == 1] = 255
    outname = 'output_images/test{0}_gf_absx_20_100.jpg'.format(i)
    scipy.misc.imsave(outname, abs_bin)
    abs_bin = gt.abs_thresh(img, orient='y', thresh=x_thresh)
    abs_bin[abs_bin == 1] = 255
    outname = 'output_images/test{0}_gf_absy_{1}_{2}.jpg'.format(i, x_thresh[0], x_thresh[1])
    scipy.misc.imsave(outname, abs_bin)
    mag_bin = gt.mag_thresh(img, thresh=(30, 100))
    mag_bin[mag_bin == 1] = 255
    outname = 'output_images/test{0}_gf_mag_30_100.jpg'.format(i)
    scipy.misc.imsave(outname, mag_bin)
    dir_bin = gt.dir_thresh(img, thresh=(0.7, 1.3))
    dir_bin[mag_bin == 1] = 255
    outname = 'output_images/test{0}_gf_dir_0p7_1p3.jpg'.format(i)
    scipy.misc.imsave(outname, dir_bin)
    i += 1
            
       
        
# ColorFilter
# read test1.jpg thru test6.jpg
# do not save since these are the same initial images from gf above
# use ColorFilter to perform gray, s, and r filters
# save each version as test*_type_thresholdvalues.jpg           
i = 1
ct = ll.ColorTransformer()    
s_thresh = (160, 255) #(90,255)
for filename in filenames:
    img = scipy.misc.imread(filename)            
    gray_bin = ct.gray_thresh(img, thresh=(180, 255))
    gray_bin[gray_bin == 1] = 255
    outname = 'output_images/test{0}_cf_gray_180_255.jpg'.format(i)
    scipy.misc.imsave(outname, gray_bin)
    s_bin = ct.s_thresh(img, thresh=s_thresh)
    s_bin[s_bin == 1] = 255
    outname = 'output_images/test{0}_cf_s_{1}_{2}.jpg'.format(i, s_thresh[0], s_thresh[1])
    scipy.misc.imsave(outname, s_bin)
    r_bin = ct.r_thresh(img, thresh=(0, 255))
    r_bin[r_bin == 1] = 255
    outname = 'output_images/test{0}_cf_r_0_255.jpg'.format(i)
    scipy.misc.imsave(outname, r_bin)
    i += 1        



# rough pipeline
# read test*.img save as test*_ppl_initial.jpg
# undistort save as test*_ppl_undistored.jpg
# apply soblex and s filters but don't save yet
# warp perspective and save as test*_ppl_warped_bin.jpg
# change filtered bin to black white for better visual appearance
# save as test*_ppl_filtered.jpg
# find lane lines using warped_bin and save as test*_ppl_lanes.jpg
for i in range(1,7):   
    img = scipy.misc.imread('test_images/test{0}.jpg'.format(i))
    scipy.misc.imsave('output_images/test{0}_ppl_initial.jpg'.format(i), img)
    undistorted = cc.undistort(img)
    scipy.misc.imsave('output_images/test{0}_ppl_undistorted.jpg'.format(i), \
                      undistorted)
    absx_bin = gt.abs_thresh(undistorted, orient='x', thresh=x_thresh)
    s_bin = ct.s_thresh(undistorted, thresh=s_thresh)
    filtered = np.zeros_like(absx_bin)
    filtered[(absx_bin == 1) | (s_bin == 1)] = 1
    warped_bin = pw.warp(filtered)
    scipy.misc.imsave('output_images/test{0}_ppl_warped_bin.jpg'.format(i), warped_bin)
    filtered[filtered == 1] = 255 # for better visuals
    scipy.misc.imsave('output_images/test{0}_ppl_filtered.jpg'.format(i), filtered)   
    warped = pw.warp(filtered)
    scipy.misc.imsave('output_images/test{0}_ppl_warped.jpg'.format(i), warped)
    llf = ll.LaneLineFinder() # reinstantiate to do full search each time
    ploty, left_fitx, right_fitx = llf.find_lane_lines(warped_bin, True)
    scipy.misc.imsave('output_images/test{0}_ppl_lanes.jpg'.format(i), llf.out_img)
    print('radii:  ({0}, {1})'.format(llf.left_rad, llf.right_rad))
    print('distance:  ({0}, {1})'.format(llf.dist_min, llf.dist_max))
            




        






    
