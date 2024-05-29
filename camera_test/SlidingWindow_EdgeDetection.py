import numpy as np
import cv2 as cv
import time
 
cap = cv.VideoCapture('/home/niklas/SeaMe/ADS01/ad_perception/camera_test/output_1.avi')

def frame_processor(image):
	warped_image = warp_image(image)
	
	# applying gaussian Blur which removes noise from the image 
	#kernel_size = 7
	# Applying gaussian blur to remove noise from the frames
	#blur = cv.GaussianBlur(warped_image, (kernel_size, kernel_size), 0)
	#cv.imshow('GaussianBlur Image', blur)

	#Using HSV filter
	frame_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	#values are tested in testing script "hsv_filter". the 3rd value can be ajusted between 150-200
	image_hsv = cv.inRange(frame_HSV, (0, 0, 80), (180, 255, 255))
	cv.imshow('hsv', image_hsv)

	#apply the sliding window for left and right lane with base midpoint of lane at xm
	left, left_line = sliding_windows(image_hsv, image, 5, xm=320)
	right, right_line = sliding_windows(image_hsv, image, 5, xm=1030)

	#calculate middlepoints with left and right lane points 
	middle_points = calculate_middle_path(left, right)

	for point in middle_points:
		draw_lines_points(image, point=point)
	return 0

def roi_boxes(image, midpoint):
	box_width = 100
	if midpoint[1]==620:
		box_width = 250
	mask = np.zeros_like(image)
	#mask_copy = mask.copy()
	start_point = (midpoint[0]-box_width, midpoint[1])
	end_point = (midpoint[0]+box_width, midpoint[1]+100)
	mask_box = cv.rectangle(mask, start_point, end_point, 255, -1) 
	masked_image = cv.bitwise_and(image, mask_box)
	cv.imshow('Box Image', mask_box)
	return masked_image

def sliding_windows(image, warped_image, num_windows, xm=320):
	height, width = image.shape
	result = [[-1,-1,-1]]*num_windows
	line = [[-1,-1,-1]]*num_windows
	midpoint = (xm, height-100)
	for i in range(num_windows):
		im_h = sobel_inner_line(image, xm)
		#cv.imshow('Sobel Image', im_h)
		masked_image = roi_boxes(im_h, midpoint)
		#cv.imshow('Masked Image', masked_image)
		lines = hough_transform(masked_image)
		if lines is not None:
			average_line, average_lane_line = average_lane_lines(lines, midpoint)
			if len(average_lane_line) <= 1:
				break
			point = (average_lane_line[2].astype(int),midpoint[1])
			result[i] = point
			line[i] = average_lane_line
			draw_lines_points(warped_image, average_line, point)
			midpoint = (point[0], height-(i+2)*100)
	return result, line

def draw_lines_points(image, lines=None, point=None):
	#draw lines and points and show the image
	draw_image = image
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line
			draw_image = cv.line(image, (int(x1),int(y1)),(int(x2),int(y2)),(0, 0, 255),5)
	if point is not None:
		draw_image = cv.circle(image, (point[0],point[1]), radius=5, color=(0, 255, 0), thickness=-1)
	cv.imshow("Hough Transformation",draw_image)

def calculate_middle_path(left, right):
	#calculate the middle of left and right lane with given parameters
	middle = [None] * 5
	half_lane_width = 520	#assumpt half lane in pixels
	for i in range(len(left)):
		x_left = left[i]
		x_right = right[i]
		if len(x_left) <=2 and len(x_right) <=2:
			#in case of both lanes detected, calculate middle 
			middle[i] = (int((x_right[0]/2 + x_left[0]/2)),x_left[1])
		elif x_left != [-1,-1,-1]:
			#if only left lane is detected, calculate middle 
			middle[i] = (x_left[0]+half_lane_width,x_left[1])
		elif x_right != [-1,-1,-1]:
			#if only right lane is detected, calculate middle 
			middle[i] = (x_right[0]-half_lane_width,x_right[1])
	return middle

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

def sobel_inner_line(image, xm):
	#apply sobel filter in x direction
	height, width = image.shape
	if xm <= width/2:	#for left lane invert the image
		img = cv.bitwise_not(image)
	else:
		img = image
	sobelx= cv.Sobel(img,cv.CV_8U,1,0,ksize=3) #only ditect inner lines
	return sobelx

def hough_transform(image):
    # Distance resolution of the accumulator in pixels.
    rho = 1			
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180
    # Only lines that are greater than threshold will be returned.
    threshold = 50
    # Line segments shorter than that are rejected.
    minLineLength = 30
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 5
    # function returns an array containing dimensions of straight lines 
    # appearing in the input image
    return cv.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
        minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_lane_lines(lines, midpoint): 
	valid_right_line = []
	valid_lines = [] #(slope, intercept)
	xm,ym = midpoint
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				if x1 == x2:
					x2 = x2+1
				if y1 == y2:
					y2 = y2+1
				# calculating slope of a line
				slope = (y2 - y1) / (x2 - x1)
				# calculating intercept of a line
				intercept = y1 - (slope * x1)
				# intercept with horizontal line
				x = (ym - intercept) /slope
				if not -0.2 < slope < 0.2:
					valid_right_line.append(line)	#useable complete line
					valid_lines.append((slope, intercept, x))	#useable slope, intercept and x value for lines
		#calculate the average of valid lines
		if len(valid_lines) > 0:
			average_line = np.mean(valid_right_line, axis=0)
			average_lane_line = np.mean(valid_lines, axis=0)

			return average_line.astype(int), average_lane_line
		else:
			return [[-1,-1,-1,-1]] , [[-1,-1,-1]]

while cap.isOpened():
	ret, frame = cap.read()

	frame_processor(frame)
	#cv.imshow('frame', frame)

	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()