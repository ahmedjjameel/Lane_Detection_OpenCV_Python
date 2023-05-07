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

