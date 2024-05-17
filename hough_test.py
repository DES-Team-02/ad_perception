# testing the hough line detection with different parameters which are ajustable with the trackbars

import cv2 as cv
import argparse
import time
import numpy as np

max_value = 255
max_value_H = 5
low_rho = 1
low_minLineLength = 0
low_V = 0
low_threshold = max_value_H
low_maxLineGap = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
rho = 'rho'
minLineLength = 'minLineLength'
threshold = 'threshold'
maxLineGap = 'maxLineGap'


## [low]
def rho_thresh_trackbar(val):
    global low_rho
    global low_threshold
    low_rho = val
    cv.setTrackbarPos(rho, window_detection_name, low_rho)
## [low]

## [high]
def threshold_thresh_trackbar(val):
    global low_rho
    global low_threshold
    low_threshold = val
    cv.setTrackbarPos(threshold, window_detection_name, low_threshold)
## [high]

def minLineLength_thresh_trackbar(val):
    global low_minLineLength
    global low_maxLineGap
    low_minLineLength = val
    cv.setTrackbarPos(minLineLength, window_detection_name, low_minLineLength)

def maxLineGap_thresh_trackbar(val):
    global low_minLineLength
    global low_maxLineGap
    low_maxLineGap = val
    cv.setTrackbarPos(maxLineGap, window_detection_name, low_maxLineGap)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

## [cap]
cap = cv.VideoCapture('/home/niklas/SeaMe/ADS01/ad_perception/camera_test/output.avi')
## [cap]

## [window]
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
## [window]

## [trackbar]
cv.createTrackbar(rho, window_detection_name , 1, 8, rho_thresh_trackbar)
cv.createTrackbar(threshold, window_detection_name , 50, 150, threshold_thresh_trackbar)
cv.createTrackbar(minLineLength, window_detection_name , 20, 150, minLineLength_thresh_trackbar)
cv.createTrackbar(maxLineGap, window_detection_name , 5, 100, maxLineGap_thresh_trackbar)
## [trackbar]

def frame_processor(image):
    
    # applying gaussian Blur which removes noise from the image 
    # and focuses on our region of interest
    # size of gaussian kernel
    kernel_size = 7
    # Applying gaussian blur to remove noise from the frames
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    #cv.imshow('GaussianBlur Image', blur)

    #Using HSV filter
    frame_HSV = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    image_hsv = cv.inRange(frame_HSV, (0, 0, 150), (180, 255, 255))
    #cv.imshow('hsv', image_hsv)

    #Using Canny Edge detection
    # first threshold for the hysteresis procedure
    low_t = 125
    # second threshold for the hysteresis procedure 
    high_t = 150
    # applying canny edge detection and save edges in a variable
    edges = cv.Canny(image_hsv, low_t, high_t)
    sobelx = cv.Sobel(image_hsv,cv.CV_8U,1,0,ksize=3)
    #cv.imshow('Canny Image', sobelx)
    return sobelx

def warp_image(image):
    #create warped image with fixes parameters
    tl = (320,110)
    bl = (0 ,420)
    tr = (800,110)
    br = (1070,420)
    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv.warpPerspective(image, matrix, (1280,720))
    #cv.imshow('warp', transformed_frame)
    return transformed_frame

while True:
    ## [while]
    ret, frame = cap.read()
    if frame is None:
        break
    warped_image = warp_image(frame)
    cv.imshow(window_capture_name, frame)
    canny = frame_processor(warped_image)
    lines = cv.HoughLinesP(canny, rho = low_rho, theta = np.pi/180, threshold = low_threshold,
        minLineLength = low_minLineLength, maxLineGap = low_maxLineGap)
    if lines is None:
        continue
    for line in lines:
        x1,y1,x2,y2 = line[0]
        hough_img = cv.line(warped_image, (x1,y1),(x2,y2),(0, 255, 0),5)
        cv.imshow("Hough Transformation",hough_img)

    time.sleep(0.5)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break