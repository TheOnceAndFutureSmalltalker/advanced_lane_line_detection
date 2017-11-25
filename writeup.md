

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

### Integration and System Testing

As mentioned earlier, after component development, I did integration and system testing.  These are fond in files integ_tests.py and sys_test.py respectively.  I started the tuning process here for each of the components.  The system test was itself a rough pipeline where I took a single image all the way through the process.  Many of the images seen so far were actually constructed at this stage.

### Pipeline

At this point I started developing the `Pipeline` class, found in lanelines.py, lines 335-491.  Instances of the previously mentioned components are injected into an instance of this class through the constructor.  The file pipeline_runner.py actually creates an instance of the pipeline and all of its components and reads and writes the video files.

The main method in `Pipeline` is `proces_image()` which takes a single raw video frame or image, runs it through all of the processing, and returns a copy of the image with the road lane clearly marked with a green mask.  It also writes the lane radii and the lane width min and max at the top of the image.  This labeling proved to be very useful in trouble shooting specific images and developing rules for determining "bad" images.  There are several helper methods in Pipeline as well.

Initially, I started with test images just to get the mechanics of the code correct.  I started with all of the tuning parameters set from the integration and system tests.  Then I captured 2 seconds worth of images from the project video and processed each of those as separate images.  Next I tried 10 seconds of actual video production to see how the initial settings were working.  I constantly went badk and forth from producing video snippits to processing individual images take from various points on the video.   

I tried several techniques for improving my solution.  I created a test for bad images and varied this test as I went and found new cases.  I started keeping a history of the most recent n images and averaging over the history to smooth out the presentation.  The Udacity lesson mentioned combining the gradient X trasnform with the S transform since each of these techniques picks up on the lane lines under different circumstances.  I used this as well.  

I tweaked several prameters to get better results.  Primarily I tweaked the source and destination points in the warping process and I tweaked the window search parameters in the lane finding component.  I did very little tweaking on the thresholds for gradient X and S transforms as the original ones worked very well from the start.  I did not try using any other gradient or color transforms as they did not look like they would provide any benefit based on the test images I produced.  

In the warping process, I felt I was going too deep - too far down the road.  The problem is, 5 pixels of depth at bottom of the image might be a half meter or so, but 5 pixels at the top might be 5 meters or more!.  So the source region is very sensitive at the top and the lane lines tend to disappear in the distance.  Shortening the region of interest made a difference.  I inspected several actual video images to tweak these points.  My final warping settings are as follows:

| Target | Points |
| Source | [178,720], [573,462], [709,462], [1135,720] |
| Destination | [337,720], [337,0], [970,0], [970,720] |

My initial test for a "bad" image was just to use a ration of min to max lane width and compare that with some threshold such as 0.85.  The idea here being that the lane width should not vary much and if it did, it would indicate a bad fit.  This didn't seem to work all too well as obviously bad images were still getting through.  I then started inspecting individual images and various trouble spots and developing specfic rules to handle these cases.  My final rules for bad images were:

1. any image whose min lane width < 500

2. any image whose min lane width < 550 AND ration of min radius to max radius is < 0.4

3. if x bad fits in a row, reset lanelineFinder component to do a complete window search with next image

The smoothing process by averaging over the last n images does smooth things out but introduces its own issues.  There were a few points in the video where the lane managed to change quite a bit form one frame to the next and the smoothing process cannot react to this in a timely fashion

At this point, I still seemed to be getting some bad behavior at the top of the image.  Also, there was a special case where the car hits a bump and the camera's angle changes causing the lane lines to go wide or narrow.  These were the hardest cases to crack and I never did fully fix them as you can see in the resulting video.  Every time the car hits a bump, the green mask jumps a bit, but settles down again after a few frames.  Such a jostling of the car always happens when the car transitions from one type of pavement to the other and this affect is quite noticeable at these points in the video.


So, my final solution was as follows:

| Step | Description |
| Undistort | Undistort the original image using `CameraCalibrato`r component |
| Gradient X | Transform a copy of the image to a binary gradient X transform using `GradientTransformer` with thresholds (20, 100) |
| Saturation | Transform a copy of the image to binary based on Saturation level using `ColorTransformer` with thresholds (150, 255) |
| Combine | Combine the X and S transforms into one using an AND operator to yield a binary image with max lane line definition |
| Warp | Use the PerspectiveWarper component and points source and destination points defined above to get a warped image |
| Lane Line | Run this image through the LaneLineFinder component with settings of nwindows=10, margin=40, minpix=25 |
| Check Fit | Apply rules above, if bad image keep last good fit, if good image use new fit. Update bad image counter. |
| History | Update history with current good fit (which may be previous fit if latest fit is bad.) |
| Create Mask | Apply lane mask to warped image using average of most recent n fits kept in history. |
| Unwarp | Unwarp the mask back to car perspective. |
| Combine Mask/image | Add mask to copy of orginal undistorted image thereby clearly marking the lanes. |
| Add Labels | Label the image with radii and lane widths. | 
| Return | Return completed image, original is untouched, and may optionally save a copy to file. |


Note:  The labeling applied in last step was that of the most recent good fit and not of the average fit that is actually applied to the image!  Some of the Pipeline parameters are hard coded and not set up as instance variables.  Especially the rules that are applied for image fit.  

Here is an example of original and completed image.

<p align="center">
<img src="" width="320px" /><br /><b>Test image taken with camera</b></p>
<br />
<p align="center">
<img src="" width="320px" /><br /><b>Test image taken with camera</b></p>
<br />

The completed video is the file is project_video_lanes.mp4 and is located <a href="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/project_video_lanes.mp4">here</a>. 

### Discussion

The main problem I had that I was never able to completely solve was the case of the car hitting a bump or transitionly from one pavement to the other causing a pitching up and down of car resulting in the lane lines in the video suddenly widening and narrowing and I would temporarily loose a good fit.  This is noticeable in the video. Perhaps a higher fraem rate would help here as the changes from frame to frame would be less substantial.

I would also like to improve on the test for bad image and how that case is handled.  The test can only be improved by seeing more examples and applying more logic.  I would like to try handling the case by rerunning the lane line finder using a full window search to see if that fixes it, and if not, only then use the previous good fit.

The averaging function could be improved by providing weights so that the most recent good fit has the most influence on the average.

Finally, I would like to improve the code.  I like the way it is factored and tested but there is still much work to be done.  Too many tunable parameters are hard coded.  I would like to have a large config object that is set up and passed in to the `Pipeline` object.  Also, the saving of intermediate images could be handled better.  In too many cases it is just hard coded.

Finally, I would like to try the challenge videos.  I have not done that yet, but will and continue tuning my solution.

Also, while the code is factored quite a bit and 
