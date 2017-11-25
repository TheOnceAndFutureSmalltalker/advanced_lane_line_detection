

# Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### General Approach

Because of the length and complexity of the programming task, I decided to write components/classes for each of the steps above and one large integration component for the pipeline itself.  I also decided to use a TDD approach to my development for each of these components, followed by integration and system tests, developing and enhancing the code as I tested.  The components/classes are listed below:
<br />

Class/Component | Description 
------|------
`CameraCalibrator` | Calibrates a camera based on a series of test imaes.  Can then remove distortions of an image taken by that cameral
`GradientTransformer` | Performs various transforms, using Sobel transform, of an image into gradient space.
`ColorTransformer` | performs various transforms of an image into various color spaces  gray, red, saturation, etc.
`PerspectiveWarper` | Warps an image based on a change in perspective from a car's view to a bird's eye view of the road.  Can also transform back.
`LaneLineFinder` | Identifies right and left lane lines of a binary warped image, returns pixel coords for those lines, maintains stats for those lines like polynomial coefficients, curvature radius, max/min lane widths.  Optionally creates an image of the lane lines for intermediate analysis.
`Pipeline` | Fully processes an image by identifying lane lines and marking the lane in green.  Information is also placed at top of image. Processed image is returned, original image is untouched.
<br />

All of these implementations are found in the file lanelines.py so it can be imported as a single library.  In addition, each of the components (except for Pipeline) has its own unit test file:
<br />

Class/Component | Unit Test File 
------|------
`CameraCalibrator` | unit_test_cam_cal.py
`GradientTransformer` | unit_test_grad_trans.py
`ColorTransformer` | unit_test_colr_trans.py
`PerspectiveWarper` | unit_test_persp_warp.py
`LaneLineFinder` | unit_test_find_lns.py
<br />

There is also an integ_test.py for integration tests among the components and sys_test.py for system testing, which is itself, a rough pipeline.  It is with these two test files that I started to tune parameters and save test images in various stages of completion.  

Finally, pipeline_runner.py was used to actually process the video images and create the finished products.  It was also used for creating sets of test images from the video itself, usually 2-5 seconds worth, for testing, inspection, and fine tuning of those particular images.

### Camera Calibration

The code for the `CameraCalibration` class is found in lanelines.py file, lines 8-47.  

I used various test chessboard images captured with the car's camera for calibrating the camera.  All of the cheesboard images were 9 vertices across by 6 down.  In the `calibrate()` method you can see that I used openCV library to read each test image, convert it to gray scale, and find the coordinates of the vertices.  I also used numpy `mgird()` method to compute the undistorted coordinates of these vertices.  For each image I appended the image vertices and calculated vertices to respective collectoins.  I then used openCV `calibrateCamera()` on these to collections to calibrate.  This method returns the matrix and distortion coefficients necessary to undistort any image taken with the camera and they are stored as instance variables.  I then wrote the undistort() method to use these values and the openCV `undistort()` method to undistort any input image and return the undistorted version.

Below are an example of a original chessboard image and its undistorted version.
<br /><br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_initial_checkerboard_image.jpg" width="320px" /><br /><b>Test image taken with camera</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_undistorted_checkerboard_image.jpg" width="320px" /><br /><b>Undistorted test image</b></p>
<br />
And next are a road image taken with the camera, and its undistorted version.
<br /><br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_initial_road_image.jpg" width="320px" /><br /><b>Road image taken with camera</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_undistorted_road_image.jpg" width="320px" /><br /><b>Undistorted road image</b></p>
<br />

### Gradient Transformation

The code for the `GradientTransformer` class is found in lanelines.py file, lines 88-129.  This component uses a Sobel transform to compute gradients for pixels in the image.  The component is initialized with kernel size and color to gray conversion type (RGB2GRAY for example).

There are 3 methods - one for absolute gradient, `abs_thresh()`, one for magnitude of gradient `mag_thresh()`, and one for gradient direction, `dir_thresh()`.  Each method takes an imput image and an upper and lower threshold value.  `abs_thresh()` also takes an orientation parameter of 'x' or 'y'.  Each method converst input image to grayscale, then uses the openCV `Sobel()` function to calculate the gradients for the image.  The `abs_thresh()` method converts all results to absolute value since we don't care about the direction of the gradient, just the value of the change in either the x or y direction.  The `mag_thresh()` method calculates the magnitude of the x and y components of the gradient vector.  The `dir_thresh()` calculates the direction of the gradient from the x and y components.  All three methods return a binary versoin of the output image. 

Below are some examples of these three gradient transforms on test road images.  In order to make these images viewable, I had to transform them from binary to black white images where the 1 pixels in binary became (255,255,255) pixels for viewing the images.
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test1_gf_absx_20_100.jpg" width="320px" /><br /><b>Absolute gradient X with threshold (20, 100)</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test1_gf_absy_20_100.jpg" width="320px" /><br /><b>Absolute gradient Y with threshold (20, 100)</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test1_gf_mag_30_100.jpg" width="320px" /><br /><b>Maginitude gradient with threshold (30, 100)</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test1_gf_dir_0p7_1p3.jpg" width="320px" /><br /><b>Directional gradient with threshold (in rads) (0.7, 1.3)</b></p>
<br />

It can be seen from this image that the absolute gradient in X directoin is a good candidate for identifying lane lines. 

### Color Transformation

The code for the `ColorTransformer` class is found in lanelines.py file, lines 133-169.  This component is initialized with the color channel of the target images - RGB or BGR. 

There are three working methods - one for inspecting gray scale, `gray_thresh()`, one for inspecting red component of RGB image, `r_thresh()`, and one for sinpecting saturation component of HSL image, `s_thresh()`.  Each method takes a target image and an upper and lower threshold for the particular component of interest.  If the component of interest, red color for example, is within the given threshold, then that pixel gets a 1,  If not, the pixel gets a value 0.  The result is a bainry image showing where that component of interest meets the threshold range.

Below are examples of thes transformations performed on a test image of the road taken form the car.  As above, in order to make these images viewable, I had to transform them from binary to black white images where the 1 pixels in binary became (255,255,255) pixels for viewing the images.
<br />

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test2_cf_gray_180_255.jpg" width="320px" /><br /><b>Gray transformation threshold (180, 255)</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test2_cf_r_200_255.jpg" width="320px" /><br /><b>R transformation threshold (200, 255)</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test2_cf_s_160_255.jpg" width="320px" /><br /><b>S transformation threshold (160, 255)</b></p>
<br />

From these examples, it looks like the S transformed image will be useful in identifying lane lines.


### Perspective Warping

In order to identify lanes in a picture, I needed a view of the road from on top so that the lane lines look like two parallel lines.  unfortunately, the road camera is looking down the road with the lane lines coming together on the horizon.  To solve this, i needed to warp the image to the bird's eye view perspective.  For this step, I created the `PerspectiveWarper` class.  This code is found in lanelines.py file, lines 51-84.  

This component has `warp()` method with takes a target image taken from viewpoint of the car looking down the road, and warps it to a destination image from the viewpoint of looking over the road from up above.  The `unwarp()` method performs the complimentary operation.  

In order to do this, I had to get a straight line road image that had already been undistorted by the `CameraCalibrator`.  


<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/pt_straight_lines_distorted.jpg" width="320px" /><br /><b>Original road image</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/pt_straight_lines_undistorted.jpg" width="320px" /><br /><b>Undistorted road iamge</b></p>
<br />

I then captured points at bottom of lane and top of lane.  These became my source points for the transform.  This shape is basically a trapezoid.  I then determined the destination points for the warped image which is basically a rectangle running the full height of the image and having same width as bottom of trapezoid in source.  The PerspectiveWarper is initialized with these two sets of points.

There are two methods - `warp()` for warping an initial image to an overhead perspective, and `unwarp()` for warping an overhead image back to road perspective.  Both methods take an input image and return the warped/unwarped image with the initial image unchanged.  Both methods use the openCV function `getPerpectiveTranform()` to calculate the transform or the reverse transform.  These transforms are then saved as instance variables so they do not have to be recalculated for subsequent images.  

Below is the image above sufficiently warped to an overhead perspective.  The pipeline image processing will not warp the image until it has been transformed to a binary state, but I am using a full road image here for better visual effect.  It is also how I tested to make sure it was working correctly.  In warping a binary image it is not quite so easy to follow what is actually going on in the warped image.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/pt_straight_lines_warped.jpg" width="320px" /><br /><b>Warped road image</b></p>
<br />

Below is the same image unwarped back to road perspective.  Notice that in the process we lost all of the original image outside of the initial trapezoid.  That's OK since we are only interested in the lane at this point and creating a mask of this section of the unwarped image to overlay on the original image (see Pipeline section below.)

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/pt_straight_lines_unwarped.jpg" width="320px" /><br /><b>Unwarped road image</b></p>
<br />

### Finding Lane Lines

Next I developed the `LaneLineFinder` component.  This code is found in the lanelines.py file, lines 173-331.  This component will find the lane lines in an image that has already been undistorted, transformed to binary format identifying edges, and warped to overhead perspective.  It returns arrays of points identifying the lane lines.  It also calculates statistics for those lane lines like polynomial coefficients for the second order polynomial that fits the lane line, the radius of each lane line, and the max and min lane width.  The input image remains untouched. Optionally, this component will create output images for testing and tuning purposes.

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test3_ppl_warped.jpg" width="320px" /><br /><b>Typical input image for LaneLineFinder</b></p>
<br />

Note: the image above has been enhanced from its binary form for viewing purposes!

The main method of this component is `find_lane_lines()` which takes the input image (similar to image above) and returns thre arrays of points identifying the 2 lane lines.  These three arrays are the y coordinates (same for both lanes) and the x coordinates of the left and right lanes respectively.  

The `find_lane_lines()` first searches the entire image for the two lane lines.  The Udacity lesson mentioned two separate search methods for this purpose.  One uses a sliding window for which a histogram of 1 pixels is determined and the highest level in the histogram is considered to be the lane line.  The second method uses a convolution to maximize the number of "hot" pixels.  I tried both and settled on the former.  Both of these methods identify points likely to contain the true lane line which are then fit to a second order polynomial to define the entire line, top to bottom.  From these polynomial equations, I calculated the pixel points for the two lines and returned these as the output. Optionally, this method will create an ouput image identifying the pixels found to be associated with the lane lines, the windows found to contain the lane lines, and the polynomial fit lines.

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test3_ppl_lanes.jpg" width="320px" /><br /><b>Optional output image showing results</b></p>
<br />

For efficiency, subsequent calls to `find_lane_lines()` will skip the exhaustive window search and start searching from the previously found set of lines.  Also, radii and max/min lane widths are calculated for the lane lines. All information for a set of lane lines (pixel points identifying the lines, polynoimial coefficents, radii, and lane widths) are stored in instance variables for access by client code.

Finally, the reset() method can be called to start a new exhaustive window search upon the next call to `find_lane_lines()`.

This component can be tuned with three key parameters:  number of windows along the distance of the lane, margin or width of windows (width=2 X margin), and number of pixels required to recenter a window.  These were used extensively in my tuning process.

Here are other examples of images in and found lane lines.

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test4_ppl_warped.jpg" width="320px" /><br /><b>Input image from test4.jpg</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test4_ppl_lanes.jpg" width="320px" /><br /><b>Output image from test4.jpg</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test5_ppl_warped.jpg" width="320px" /><br /><b>Input image from test5.jpg</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/test5_ppl_lanes.jpg" width="320px" /><br /><b>Output image from test5.jpg</b></p>
<br />

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
