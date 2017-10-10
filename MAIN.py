# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import math
imageio.plugins.ffmpeg.download()
import os
os.listdir("test_images/")


#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

def process_image(image):

#printing out some stats and plotting
        print('This image is:', type(image), 'with dimensions:', image.shape)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mpimg.imsave("Test_images_output/solidWhiteCurve_Gray.png", gray)
# Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        mpimg.imsave("Test_images_output/Gaussian.jpg", blur_gray)
# Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        mpimg.imsave("Test_images_output/Edges.jpg", edges)
        
# Next we'll create a masked edges image using cv2.fillPoly()
#def region_of_interest(img, vertices):
#defining a blank mask to start with

        mask = np.zeros_like(edges)

        mpimg.imsave("Test_images_output/Mask.jpg", mask)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
                channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,) * channel_count
        else:
                ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
        imshape = image.shape
        vertices = np.array([[(100,imshape[0]), (440, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #mpimg.imsave("Test_images_output/whiteCarLaneSwitch_masked_edges.png", masked_edges)
    
    #returning the image only where mask pixels are nonzero
        masked_edges = cv2.bitwise_and(edges, mask)
        mpimg.imsave("Test_images_output/Mask_Edges.jpg", masked_edges)
#   return masked_image
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 20   # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100 #minimum number of pixels making up a line
        max_line_gap = 200    # maximum gap in pixels between connectable line segments
        line_image = np.copy(image)*0 # creating a blank to draw lines on

        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
   
# Create a "color" binary image to combine with line image
#color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
        lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        mpimg.imsave("Test_images_output/solidWhiteCurve_Lines_Edges.png", lines_edges)
#    plt.imshow(lines_edges)

# Display the image
#mpimg.imsave("Test_images_output/whiteCarLaneSwitch_lines_edges.png", line_edges)



    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)


        return lines_edges


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML



white_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4").subclip(0,3)
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)



