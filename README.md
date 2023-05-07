## Lane Detection using OpenCV and Python

### Overview

Detect lanes using computer vision techniques. In this project, I used Python and OpenCV to detect lane lines on the road. I developed a processing pipeline that works on a series of individual images, and applied the result to a video stream.

![solidWhiteRight_output](https://user-images.githubusercontent.com/81799459/236697500-a4190b06-e3ce-4cdc-b203-fe8d0b845725.gif)

### Pipeline architecture:
1.	Load test images.
2.	Apply Color Selection
3.	Apply Canny edge detection.
o	Apply gray scaling to the images.
o	Apply Gaussian smoothing.
o	Perform Canny edge detection.
4.	Determine the region of interest.
5.	Apply Hough transform.
6.	Average and extrapolating the lane lines.
7.	Apply on video streams.

