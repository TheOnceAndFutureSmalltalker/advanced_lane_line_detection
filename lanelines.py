import numpy as np
import cv2
import glob
import scipy.misc



class CameraCalibrator:
    """Calibrates a camera based on a series of test images.
       Can then remove distortions of an image taken by that camera.
    """

    def __init__(self, nx, ny, file_template):
        self.nx = nx
        self.ny = ny
        self.file_template = file_template
        self.mtx = None
        self.dist = None
 
    def calibrate(self):
        objpoints = []
        imgpoints = []
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        filenames = glob.glob(self.file_template)
        gray = None
        for filename in filenames:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
                #cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
        assert len(objpoints) == len(imgpoints) 
        #print(gray.shape)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera( \
                            objpoints, imgpoints, \
                            gray.shape[::-1], None, None)            

    def undistort(self, img):
        assert type(img) is np.ndarray and len(img.shape) >= 2 ,\
        'img must be numpy array of at least 2 dimensions'
        if self.mtx is None:
            self.calibrate()
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst
    
    
    
class PerspectiveWarper:
    """warps an image based on a change in perspective from car's view.
       to a birds-eye view.
    """
    
    def __init__(self, src, dst):
        assert type(src) is np.ndarray and src.shape == (4,2) ,\
        'src must be numpy array of shape (4, 2) of floats'
        assert type(dst) is np.ndarray and dst.shape == (4,2) ,\
        'dst must be numpy array of shape (4, 2) of floats'
        self.src = src
        self.dst = dst
        self.transform = None
        self.reverse_transform = None
        
    def warp(self, img):
        assert type(img) is np.ndarray and len(img.shape) >= 2 ,\
        'img must be numpy array of 2 or 3 dimensions'
        if self.transform is None:
            self.transform = cv2.getPerspectiveTransform(self.src, self.dst)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.transform, img_size, \
                                     flags=cv2.INTER_NEAREST)
        return warped
    
    def unwarp(self, img):
        assert type(img) is np.ndarray and len(img.shape) >= 2 ,\
        'img must be numpy array of 2 or 3 dimensions'
        if self.reverse_transform is None:
            self.reverse_transform = cv2.getPerspectiveTransform(self.dst, self.src)
        img_size = (img.shape[1], img.shape[0])
        unwarped =  cv2.warpPerspective(img, self.reverse_transform, \
                                        img_size, flags=cv2.INTER_NEAREST)
        return unwarped
    
    
     
class GradientTransformer:
    """Performs various transforms of an image into gradient space.""" 
    
    def __init__(self, kernel=3, color_convert=cv2.COLOR_RGB2GRAY):
        self.kernel = kernel
        self.color_convert = color_convert
        
    # from exercise thresh=(20,100)
    def abs_thresh(self, img, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(img, self.color_convert)
        if(orient == 'y'):
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        abs_binary = np.zeros_like(scaled_sobel)
        abs_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return abs_binary

    # from exercise thresh=(30,100)
    def mag_thresh(self, img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, self.color_convert)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel)
        sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return mag_binary
    
    # from exercise kernel=15 thresh=(0.7,1.3)  
    def dir_thresh(self, img, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, self.color_convert)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel)
        sobelx_abs = np.absolute(sobelx)
        sobely_abs = np.absolute(sobely)
        direction = np.arctan2(sobely_abs, sobelx_abs)
        dir_binary = np.zeros_like(direction)
        dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return dir_binary     
    
    
   
class ColorTransformer:
    """Performs various transforms of an image into color space."""
    
    def __init__(self, color_channel='RGB'):
        assert color_channel == 'RGB' or color_channel == 'BGR', \
        'color_channel must be RGB or BGR'
        self.color_channel=color_channel
            
    # from exercise thresh=(180,255)
    def gray_thresh(self, img, thresh=(0, 255)):
        if self.color_channel == 'RGB':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_binary = np.zeros_like(gray)
        gray_binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1 
        return gray_binary

    # from exercise thresh=(90,255)
    def s_thresh(self, img, thresh=(0, 255)):
        if self.color_channel == 'RGB':
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        else:
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s = hls[:,:,2]
        s_binary = np.zeros_like(s)
        s_binary[(s > thresh[0]) & (s <= thresh[1])] = 1
        return s_binary
    
    # from exercise kernel=15 thresh=(200,255)  
    def r_thresh(self, img, thresh=(0, 255)):
        if self.color_channel == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = img[:,:,0]
        r_binary = np.zeros_like(r)
        r_binary[(r > thresh[0]) & (r <= thresh[1])] = 1
        return r_binary



class LaneLineFinder:
    """Identifies right and left lane lines of a binary warped image, returns
       pixel coords for those lines, maintains stats like polynomial coefficients
       of the lines, max and min distance between lines, radius of both lines
       optionally creates an image of the lane lines.
    """
    
    def __init__(self, nwindows=9, margin=80, minpix=50):
        self.nwindows = nwindows # number of windows top to bottom
        self.margin = margin  # window width
        self.minpix = minpix  # recenter square if found more than many pixels
        self.out_img = None   # output image, if requested
        self.left_fit = None  # left polynomial coefficients
        self.right_fit = None # right polynomial coefficients
        self.left_rad = None  # left radius in meters
        self.right_rad = None # right radius in meters
        self.ploty = None     # y pixels for lane lines
        self.left_fitx = None # x pixels for left lane line
        self.right_fitx = None # x pixels for right lane line
        self.dist_max = None  # max distance between lane lines
        self.dist_min = None  # min distance between lane lines
        self.last_img = None  # last image processed
        self.center_offset = None # meters car is from center of lane
        

    def reset(self):
        """Set it up so next call to find_lane_lines does full window search"""
        self.left_fit = None
        self.right_fit = None        


    def find_lane_lines(self, img, create_out_img=False):
        """Returns (ploty, left_fitx, left_fity).
           creating an out_img is optional.
        """
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Create an output image to draw on and visualize the result
        if create_out_img == True:
            self.out_img = np.dstack((img, img, img))*255
        else:
            self.out_img = None        
        # create lane indices of pixels in lane
        if self.left_fit is None:
            # For first image, use histogram and sliding window to find lanes
            left_lane_inds = []
            right_lane_inds = []
            # Take a histogram of the bottom half of the image
            img_h = img.shape[0]
            histogram = np.sum(img[int(0.75*img_h):,:], axis=0)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Set height of windows
            window_height = np.int(img_h/self.nwindows)
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Step through the windows one by one
            for window in range(self.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = img_h - (window+1)*window_height
                win_y_high = img_h - window*window_height
                win_xleft_low = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                win_xright_low = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                # Draw the windows on the visualization image
                if create_out_img == True:
                    cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                    cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > self.minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > self.minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))  
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)            
        else:
            # for subsequent images, use fit as approximation of lane location
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
            self.left_fit[2] - self.margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
            self.left_fit[1]*nonzeroy + self.left_fit[2] + self.margin))) 
            
            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
            self.right_fit[2] - self.margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
            self.right_fit[1]*nonzeroy + self.right_fit[2] + self.margin)))  

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # calculate points for identified lane lines
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
   
        # Calculate radius of lane lines
        ym_per_pix = 35/img.shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(img[0])
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        self.left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.dist_min = np.min(right_fitx - left_fitx)
        self.dist_max = np.max(right_fitx - left_fitx)
        
        # calculate offset of camera from center of lane in meters
        # left of center is negative, right of center is positive
        lane_center_pixels = ( left_fitx[img.shape[0]-1] + right_fitx[img.shape[0]-1] ) / 2.0
        image_center_pixels = img.shape[1] / 2
        offset_pixels = image_center_pixels - lane_center_pixels
        self.center_offset =  offset_pixels * 3.7 / 700.0

        # for output image, draw left and right lane lines in red and greed resp.
        # draw polynomial fit of lanes in yellow 
        if create_out_img == True:
            self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]    
            pts = np.int32([np.stack((left_fitx, ploty), axis=-1)])
            cv2.polylines(self.out_img, pts, False, (255,255,0), 3)
            pts = np.int32([np.stack((right_fitx, ploty), axis=-1)])
            cv2.polylines(self.out_img, pts, False, (255,255,0), 3)
            
        # cache values for further processing
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.last_img = img

        return (ploty, left_fitx, right_fitx)
    

    def lane_mask(self):
        """Returns a warped lane mask identifying the entire lane in green.
           it is up to the client object to unwarp and apply to original image.
        """
        assert self.last_img is not None, 'An image has not been processed yet.'
        warped_zero = np.zeros_like(self.last_img).astype(np.uint8)
        warped_color = np.dstack((warped_zero, warped_zero, warped_zero))
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(warped_color, np.int_([pts]), (0,255, 0))
        return warped_color



class Pipeline():
    """Fully processes an image by identifying lane lines and marking 
       the lane in green.  Information is also placed at top of image.
       Processed image is returned, original image is untouched.
    """
    # inject the various components for the pipeline
    def __init__(self, cc, gt, ct, pw, llf):
        """ Inject main components for processing and initialize all
            other instance variables.
        """
        self.i = 0      # index of frame number
        self.cc = cc    # a CameraCalibrator
        self.gt = gt    # a GradientTransformer
        self.ct = ct    # a ColorTransformer
        self.pw = pw    # a PerspectiveWarper
        self.llf = llf  # a LaneLineFinder
        self.x_thresh = (20, 100)   # Sobel x gradient thresholds
        self.s_thresh = (150, 255)  # S channel thresholds
        # maintain measures most recent good fit
        self.right_rad = None  
        self.left_rad = None
        self.dist_max = None
        self.dist_min = None
        self.ploty = None
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.center_offset = None
        # next three vars for history of last n good fits
        self.ploty_hist = None
        self.left_fitx_hist = None
        self.right_fitx_hist = None
        self.bad_frames = 0         # counts bad frames in a row
        self.num_hist = 5           # how much history to maintain
        # max allowable bad frames starting a new window search
        self.max_bad_frames = 2  
        # max and min lane widths should not diverge more than this
        self.min_lane_width_ratio = 0.8
        self.total_bad = 0
        
        
    def process_image(self, original):
        """Fully processes an image returning copy of origial with lane
           identified by a green mask.
        """
        self.i += 1
#        scipy.misc.imsave('frames_in/test{0}.jpg'.format(self.i), original)
        undistorted = self.cc.undistort(original)
        absx_bin = self.gt.abs_thresh(undistorted, orient='x', thresh=self.x_thresh)
        s_bin = self.ct.s_thresh(undistorted, thresh=self.s_thresh)
        filtered_bin = np.zeros_like(absx_bin)
        filtered_bin[(absx_bin == 1) | (s_bin == 1)] = 1
        warped_bin = self.pw.warp(filtered_bin)
        self.llf.find_lane_lines(warped_bin, False)
        self.check_fit()
        self.add_to_history()
        warped_lane_mask = self.lane_mask(warped_bin)
        unwarped_lane_mask = self.pw.unwarp(warped_lane_mask)
        final = cv2.addWeighted(undistorted, 1, unwarped_lane_mask, 0.3, 0)
        self.label_image(final)
        scipy.misc.imsave('frames_out/test{0}.jpg'.format(self.i), final) 
#        scipy.misc.imsave('pipeline_test/test{0}_out.jpg'.format(self.i), final) 
        return final
  
    
    def check_fit(self):
        """Check new fit for goodness and either keep it oif it fails, 
           replace it with previous good fit.
           The test is that the 2 lane lines do not diverge too much.
        """
        radii_ratio = min(self.llf.left_rad, self.llf.right_rad) / max(self.llf.left_rad, self.llf.right_rad)
        
#        if self.llf.dist_min / self.llf.dist_max < self.min_lane_width_ratio:
        if (self.llf.dist_min < 500.0) or ((self.llf.dist_min < 550.0) and (radii_ratio < 0.4)):
#            print('{0} bad {1:8.2f} {2:8.2f}'.format(self.i, self.llf.dist_min, radii_ratio))
            self.bad_frames += 1
            self.total_bad += 1
            # load previous fit values back into llf 
            self.llf.ploty = self.ploty
            self.llf.left_fitx = self.left_fitx
            self.llf.right_fitx = self.right_fitx
            self.llf.left_fit = self.left_fit
            self.llf.right_fit = self.right_fit
            self.llf.dist_max = self.dist_max
            self.llf.dist_min = self.dist_min
            self.llf.left_rad = self.left_rad
            self.llf.right_rad = self.right_rad
            self.llf.center_offset = self.center_offset
        else:
#            print('{0} good'.format(self.i))
            self.bad_frames = 0
            # capture current llf values as next good fit
            self.ploty = self.llf.ploty
            self.left_fitx = self.llf.left_fitx
            self.right_fitx = self.llf.right_fitx
            self.left_fit = self.llf.left_fit
            self.right_fit = self.llf.right_fit
            self.dist_max = self.llf.dist_max
            self.dist_min = self.llf.dist_min
            self.left_rad = self.llf.left_rad
            self.right_rad = self.llf.right_rad
            self.center_offset = self.llf.center_offset
            
        # if too many bad frames in a row, have llf do new window search    
        if self.bad_frames == self.max_bad_frames:
            self.llf.reset()
            self.bad_frames = 0
#            print('resetting')
            
            
    def add_to_history(self):
        """Add current fit to history."""
        if self.left_fitx_hist is None:
            self.left_fitx_hist = np.expand_dims(self.left_fitx, axis=0)
            self.right_fitx_hist = np.expand_dims(self.right_fitx, axis=0)
            self.ploty_hist = np.expand_dims(self.ploty, axis=0)
        else:
            if len(self.left_fitx_hist) == self.num_hist:
                self.left_fitx_hist = self.left_fitx_hist[1:]
                self.right_fitx_hist = self.right_fitx_hist[1:]
                self.ploty_hist = self.ploty_hist[1:]
            self.left_fitx_hist = np.concatenate((self.left_fitx_hist, [self.left_fitx]), axis=0)
            self.right_fitx_hist = np.concatenate((self.right_fitx_hist, [self.right_fitx]), axis=0)
            self.ploty_hist = np.concatenate((self.ploty_hist, [self.ploty]), axis=0)       
    
        
    def lane_mask(self, img):
        """Returns a warped lane mask identifying the entire lane in green."""
        # calculate average over current history
        left_fitx = np.average(self.left_fitx_hist, axis=0)
        right_fitx = np.average(self.right_fitx_hist, axis=0)
        ploty = np.average(self.ploty_hist, axis=0)
        
        warped_zero = np.zeros_like(img).astype(np.uint8)
        warped_color = np.dstack((warped_zero, warped_zero, warped_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(warped_color, np.int_([pts]), (0,255, 0))
        return warped_color
    
    
    def label_image(self, final):
        """Write radii, min/max lane widths, and center offset at top of image.
           Note: these are of most recent good fit and not of the average
           fit that is actually written to screen!
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'center offset: {0:8.2f}'.format(self.llf.center_offset)
        cv2.putText(final, text, (50, 50), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        text = 'radii: ({0:8.2f}, {1:8.2f})'.format(self.llf.left_rad, self.llf.right_rad)
        cv2.putText(final, text, (350, 50), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        text = 'lane width (min, max): ({0:8.2f}, {1:8.2f})'.format(self.llf.dist_min, self.llf.dist_max)
        cv2.putText(final, text, (730, 50), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)


    def save_image(self, original):
        """Saves image from video to a folder for testing."""
        self.i += 1
        scipy.misc.imsave('frames_in/test{0}.jpg'.format(self.i), original)
        return original

 
    

class OldLaneFinder:
    
    def __init__(self, window_width, window_height, margin):
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        
       
    
    def find_centroids(self, img):
        window_centroids = [] 
        window = np.ones(self.window_width) 
        offset = self.window_width/2
        img_w = img.shape[1]
        img_h = img.shape[0]
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(img[int(3*img_h/4):,:int(img_w/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum)) - offset
        r_sum = np.sum(img[int(3*img_h/4):,int(img_w/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum)) - offset+int(img_w/2)
        
        window_centroids.append((l_center,r_center)) # may not want to do this
        
        for level in range(1,(int)(img_h/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(img[int(img_h-(level+1)*self.window_height):int(img_h-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,img_w))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,img_w))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            window_centroids.append((l_center,r_center))
        
        return window_centroids

        