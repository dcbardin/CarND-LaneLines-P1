# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I import the respective packages and load a test image. Matplotlib read iage as RGB and cv2 read the iage as GBR. 

![alt text][image1]


### 2. Convert image to gray and remove the noises


To remove the noises from the gray image, it was appled cv2.GaussianBlur

[image1]: ./test_images/Gaussian.png "Gassian"

### 3. Apply Canny edges

Canny edges method is applied t find all the lines in images Without noise, the edges of lane line is very clear.


### 4. Apply Canny edges

Canny edges method is applied t find all the

[image2]: ./test_images/Edges.png "Canny on original image withut noise"




### 5. Select intereste region
In this case, the lane lines edges is located in a trapezoid. It was defined a trapezoid to restrict the region of interest.

[image3]: ./test_images/Mask_Edges.png "Region of interest"

### 6. Generate HoughLines

Function used to remove short and useless edges (Input parameter)

### 7. Apply lines on the Image
In this case, the lane lines edges is located in a trapezoid. It was defined a trapezoid to restrict the region of interest.

[image4]: ./test_images/Lines_Edges.png "Region of interest"




