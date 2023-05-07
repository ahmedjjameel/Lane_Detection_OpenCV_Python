## Lane Detection using OpenCV and Python

### Overview
Lane detection is a developing technology that is implemented in vehicles to enable autonomous navigation. Lanes detection using computer vision techniques is a very important step as the car moves along the road, it will need to know exactly where the lanes are, so it knows the boundaries and can obtain information on how to steer the car. In this project, Python and OpenCV will be used to detect lane lines on the road. The  processing pipeline is developed that works on a series of individual images, and applied the result to a video stream.

![solidWhiteRight_output](https://user-images.githubusercontent.com/81799459/236697500-a4190b06-e3ce-4cdc-b203-fe8d0b845725.gif)

### Pipeline architecture:
1.	Load test images.
2.	Apply Color Selection
3.	Apply Canny edge detection.
    -	Apply gray scaling to the images.
    -	Apply Gaussian smoothing.
    -	Perform Canny edge detection.
4.	Determine the region of interest.
5.	Apply Hough transform.
6.	Average and extrapolating the lane lines.
7.	Apply on video streams.

### Dependencies:
1.	Python 3.7.5
2.	NumPy 1.18.5
3.	OpenCV-Python 4.7.0.72
4.	Matplotlib 4.4.0
5.	Pickle5 0.0.12

### 1. Loading test images:


### 2. Color Selection:
Lane lines in the test images are in white and yellow. We need to choose the most suitable color space, that clearly highlights the lane lines. I applied color selection to the original RGB images, HSV images, and HSL images, and found out that using HSL will be the best color space to use.


### 3. Canny Edge Detection
We need to detect edges in the images to be able to correctly detect lane lines. The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. The Canny algorithm involves the following steps:
-	Gray scaling the images: The Canny edge detection algorithm measures the intensity gradients of each pixel. So, we need to convert the images into gray scale in order to detect edges.
-	Gaussian smoothing: Since all edge detection results are easily affected by image noise, it is essential to filter out the noise to prevent false detection caused by noise. To smooth the image, a Gaussian filter is applied to convolve with the image. This step will slightly smooth the image to reduce the effects of obvious noise on the edge detector.
-	Find the intensity gradients of the image.
-	Apply non-maximum suppression to get rid of spurious response to edge detection.
-	Apply double threshold to determine potential edges.
-	Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges. If an edge pixel’s gradient value is higher than the high threshold value, it is marked as a strong edge pixel. If an edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, it is marked as a weak edge pixel. If an edge pixel's value is smaller than the low threshold value, it will be suppressed. The two threshold values are empirically determined and their definition will depend on the content of a given input image.


### 4. Region of interest
We're interested in the area facing the camera, where the lane lines are found. So, we'll apply region masking to cut out everything else.



### 5. Hough Transform
The Hough transform is a technique which can be used to isolate features of a particular shape within an image.
In the Cartesian coordinate system, we can represent a straight line as y = mx + b by plotting y against x. However, we can also represent this line as a single point in Hough space by plotting b against m. For example, a line with the equation y = 2x + 1 may be represented as (2, 1) in Hough space.
 
Now, what if instead of a line, we had to plot a point in the Cartesian coordinate system. There are many possible lines which can pass through this point, each line with different values for parameters m and b. For example, a point at (2, 12) can be passed by y = 2x + 8, y = 3x + 6, y = 4x + 4, y = 5x + 2, y = 6x, and so on. These possible lines can be plotted in Hough space as (2, 8), (3, 6), (4, 4), (5, 2), (6, 0). Notice that this produces a line of m against b coordinates in Hough space.
 
Whenever we see a series of points in a Cartesian coordinate system and know that these points are connected by some line, we can find the equation of that line by first plotting each point in the Cartesian coordinate system to the corresponding line in Hough space, then finding the point of intersection in Hough space. The point of intersection in Hough space represents the m and b values that pass consistently through all of the points in the series.
 
Since our frame passed through the Canny Detector may be interpreted simply as a series of white points representing the edges in our image space, we can apply the same technique to identify which of these points are connected to the same line, and if they are connected, what its equation is so that we can plot this line on our frame.
For the simplicity of explanation, we used Cartesian coordinates to correspond to Hough space. However, there is one mathematical flaw with this approach: When the line is vertical, the gradient is infinity and cannot be represented in Hough space. To solve this problem, we will use Polar coordinates instead. The process is still the same just that other than plotting m against b in Hough space, we will be plotting r against θ.
 
For example, for the points on the Polar coordinate system with x = 8 and y = 6, x = 4 and y = 9, x = 12 and y = 3, we can plot the corresponding Hough space.
 
We see that the lines in Hough space intersect at θ = 0.925 and r = 9.6. Since a line in the Polar coordinate system is given by r = xcosθ + ysinθ, we can induce that a single line crossing through all these points is defined as 9.6 = xcos0.925 + ysin0.925.
Generally, the more curves intersecting in Hough space means that the line represented by that intersection corresponds to more points. For our implementation, we will define a minimum threshold number of intersections in Hough space to detect a line. Therefore, Hough transform basically keeps track of the Hough space intersections of every point in the frame. If the number of intersections exceeds a defined threshold, we identify a line with the corresponding θ and r parameters.
We apply Hough Transform to identify two straight lines — which will be our left and right lane boundaries



### 6. Averaging and extrapolating the lane lines
We have multiple lines detected for each lane line. We need to average all these lines and draw a single line for each lane line. We also need to extrapolate the lane lines to cover the full lane line length.


### 7. Apply on video streams
Now, we'll use the above functions to detect lane lines from a video stream. The video inputs are in test_videos folder. The video outputs are generated in output_videos folder.





