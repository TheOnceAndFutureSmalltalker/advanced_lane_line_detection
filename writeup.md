

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

The code for the `CameraCalibration` class is found in lanelines.py, lines 8-47.  

I used various test chessboard images captured with the car's camera for calibrating the camera.  All of the cheesboard images were 9 vertices across by 6 down.  In the `calibrate()` method you can see that I used openCV library to read each test image, convert it to gray scale, and find the coordinates of the vertices.  I also used numpy mgird() method to compute the undistorted coordinates of these vertices.  For each image I appended the image vertices and calculated vertices to respective collectoins.  I then used openCV.calibrateCamera() on these to collections to calibrate.  This method returns the matrix and distortion coefficients necessary to undistort any image taken with the camera and they are stored as instance variables.  I then wrote the undistort() method to use these values and the openCV.undistort() method to undistort any input image and return the undistorted version.

Below are an example of a original chessboard image and its undistorted version.
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_initial_checkerboard_image.jpg" width="320px" /><br /><b>Test image taken with camera</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_undistorted_checkerboard_image.jpg" width="320px" /><br /><b>Undistorted test image</b></p>
<br />
And next are a road image taken with the camera, and its undistorted version.
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_initial_road_image.jpg" width="320px" /><br /><b>Road image taken with camera</b></p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/advanced_lane_line_detection/blob/master/output_images/cc_undistorted_road_image.jpg" width="320px" /><br /><b>Undistorted road image</b></p>
<br />


#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

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
