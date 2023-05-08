## Lane Detection using OpenCV and Python

### Overview
Lane Detection is a computer vision task that involves identifying the boundaries of driving lanes in a video or image of a road scene. The goal is to accurately locate and track the lane markings in real-time, even in challenging conditions such as poor lighting, glare, or complex road layouts.

Lane detection is an important component of advanced driver assistance systems (ADAS) and autonomous vehicles, as it provides information about the road layout and the position of the vehicle within the lane, which is crucial for navigation and safety. The algorithms typically use a combination of computer vision techniques, such as edge detection, color filtering, and Hough transforms, to identify and track the lane markings in a road scene.

It is well known that lane recognition on freeways is an essential part of any successful autonomous driving system. An autonomous car consists of an extensive sensor system and several control modules. The most critical step to robust autonomous driving is to recognize and understand the surroundings. However, it is not enough to identify obstacles and understand the geometry around a vehicle. Camera-based Lane detection can be an essential step towards environmental awareness. It enables the car to be correctly positioned within the lane, which is crucial for every exit and the back lane - planning decision. Therefore, camera-based accurate lane detection with real-time edge detection is vital for autonomous driving and avoiding traffic accidents. In this project, Python and OpenCV will be used to detect lane lines on the road. The  processing pipeline is developed that works on a series of individual images, and applied the result to a video stream.

![solidWhiteRight](https://user-images.githubusercontent.com/81799459/236739560-a84de16e-98f0-4146-9a94-bd5935d847b0.gif)  |  ![solidWhiteRight_output](https://user-images.githubusercontent.com/81799459/236697500-a4190b06-e3ce-4cdc-b203-fe8d0b845725.gif)
:-------------------------:|:-------------------------:
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
A group of test images will be shown using a function called list_images() that shows all the test images using matplotlib.

![02  solidWhiteRight](https://user-images.githubusercontent.com/81799459/236745352-8b92cdf7-2c05-44a5-92c8-2989ea42a8a0.jpg)  |  ![04  solidYellowCurve2](https://user-images.githubusercontent.com/81799459/236745363-649b5f15-6da7-43c0-9f63-71e56e6b9471.jpg)
:-------------------------:|:-------------------------:

### 2. Color Selection:
Lane lines in the test images are in white and yellow. We need to choose the most suitable color space, that clearly highlights the lane lines. I applied color selection to the original RGB images, HSV images, and HSL images, and found out that using HSL will be the best color space to use.

![colorselect1](https://user-images.githubusercontent.com/81799459/236746496-fde89020-1467-49fb-aba6-a2b5f138248b.png)  |   ![colorselect2](https://user-images.githubusercontent.com/81799459/236746508-6556e41a-6f98-4fb5-841a-4f70f28493d9.png)
:-------------------------:|:-------------------------:

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

A Hough transform is used in the end to get the two lanes out of the image of edges. The Hough transform is a mathematical function that is used to find the lines in an image. It transforms all lines into points in the Hough space, and points into lines, and these points are calculated by the following equation. 

            ρ=x∙cos⁡(θ)+y∙sin⁡(θ)             (1)

Observing the curves produced in the Hough space will show that there are two major clusters of points of intersection in the Hough space, representing all the lines that comprise the two major lanes on either side of the road. These lines have been plotted onto the original image as illustrated in Fig. 1. 
The image shows many lines on the left lane and on the right. Each of these groups needs to be considered separately, but the number of lines must be reduced to one each. This can be achieved by taking the average of each set of Hough lines produced. This leaves a final image as illustrated in Fig. 2 that considers the average length and direction of each line. These lines may be extrapolated to account for areas of the road where there are no lanes. 

![Fig  1](https://user-images.githubusercontent.com/81799459/236732759-c958fc28-8342-4391-9e5f-8889ade81111.png)   |   ![Fig  2](https://user-images.githubusercontent.com/81799459/236732765-34b88466-e6c0-4be6-9e09-2690b3ab8525.png)
:-------------------------:|:-------------------------:

### 6. Averaging and extrapolating the lane lines
We have multiple lines detected for each lane line. We need to average all these lines and draw a single line for each lane line. We also need to extrapolate the lane lines to cover the full lane line length.


### 7. Apply on video streams
Now, we'll use the above functions to detect lane lines from a video stream. The video inputs are in test_videos folder. The video outputs are generated in output_videos folder.





