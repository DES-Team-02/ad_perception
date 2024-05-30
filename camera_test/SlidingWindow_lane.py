import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture('/home/niklas/SeaMe/ADS01/ad_perception/camera_test/output_1.avi')

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
	cv.imshow('hsv', image_hsv)
	image_roi = region_selection(image_hsv)

	#Using Canny Edge detection
	# first threshold for the hysteresis procedure
	low_t = 125
	# second threshold for the hysteresis procedure 
	high_t = 150
	# applying canny edge detection and save edges in a variable
	edges = cv.Canny(image_roi, low_t, high_t)
	sobelx = cv.Sobel(image_hsv,cv.CV_8U,1,0,ksize=3)
	cv.imshow('Canny Image', sobelx)
	return sobelx

def region_selection(image):
	# create an array of the same size as of the input image 
	mask = np.zeros_like(image) 
	# color of the mask polygon (white)
	ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0.0, rows * 1]
	top_left	 = [cols * 0.0, rows * 0.8]
	bottom_right = [cols * 1, rows * 1]
	top_right = [cols * 1, rows * 0.8]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
	test = cv.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
	masked_image = cv.bitwise_and(image, mask)
	cv.imshow('Loaded Image', test)
	return masked_image

def average_lane_lines(lines):
	valid_right_line = []
	valid_left_line = []
	
	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 == x2:
				continue
			# calculating slope of a line
			slope = (y2 - y1) / (x2 - x1)
			# slope of left lane is negative and for right lane slope is positive
			if not -1 < slope < 1: 
				if x1 > 640:
					valid_left_line.append(line)
				elif x1 < 640:
					valid_right_line.append(line)
	#
	average_right_line = np.mean(valid_right_line, axis=0)
	average_left_line = np.mean(valid_left_line, axis=0)
	return np.rint(average_left_line), np.rint(average_right_line)


def average_slope_intercept(lines):
	"""
	Find the slope and intercept of the left and right lanes of each image.
	Parameters:
		lines: output from Hough Transform
	"""
	left_lines = [] #(slope, intercept)
	left_weights = [] #(length,)
	right_lines = [] #(slope, intercept)
	right_weights = [] #(length,)
	valid_right_line = []
	valid_left_line = []
	
	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 == x2:
				continue
			# calculating slope of a line
			slope = (y2 - y1) / (x2 - x1)
			# calculating intercept of a line
			intercept = y1 - (slope * x1)
			# calculating length of a line
			length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
			# slope of left lane is negative and for right lane slope is positive
			if slope < -0.3 and x1 > 640:
				left_lines.append((slope, intercept))
				left_weights.append((length))
				valid_left_line.append(line)
			elif slope > 0.3 and x1 < 640:
				right_lines.append((slope, intercept))
				right_weights.append((length))
				valid_right_line.append(line)
	#
	average_right_line = np.mean(valid_right_line, axis=0)
	average_left_line = np.mean(valid_left_line, axis=0)

	left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
	right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
	return left_lane, right_lane

def pixel_points(y1, y2, line):
	"""
	Converts the slope and intercept of each line into pixel points.
		Parameters:
			y1: y-value of the line's starting point.
			y2: y-value of the line's end point.
			line: The slope and intercept of the line.
	"""
	if line is None:
		return None
	slope, intercept = line
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
	"""
	Create full lenght lines from pixel points.
		Parameters:
			image: The input test image.
			lines: The output lines from Hough Transform.
	"""
	left_lane, right_lane = average_slope_intercept(lines)
	y1 = image.shape[0]
	y2 = y1 * 0.6
	left_line = pixel_points(y1, y2, left_lane)
	right_line = pixel_points(y1, y2, right_lane)
	return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=4):
	"""
	Draw lines onto the input image.
		Parameters:
			image: The input test image (video frame in our case).
			lines: The output lines from Hough Transform.
			color (Default = red): Line color.
			thickness (Default = 12): Line thickness. 
	"""
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv.line(line_image, *line, color, thickness)
	return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def hough_transform(image):
    # Distance resolution of the accumulator in pixels.
    rho = 1			
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180
    # Only lines that are greater than threshold will be returned.
    threshold = 50
    # Line segments shorter than that are rejected.
    minLineLength = 10
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 25
    # function returns an array containing dimensions of straight lines 
    # appearing in the input image
    return cv.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
        minLineLength = minLineLength, maxLineGap = maxLineGap)


def warp_image(image):
    tl = (320,110)
    bl = (0 ,420)
    tr = (800,110)
    br = (1070,420)
    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv.warpPerspective(frame, matrix, (1280,720))
    cv.imshow('warp', transformed_frame)
    return transformed_frame

while cap.isOpened():
	ret, frame = cap.read()

	#cv.imshow('frame', frame)
	warped_image = warp_image(frame)
	canny = frame_processor(warped_image)
	lines = hough_transform(canny)
	if lines is None:
		continue
	result = draw_lane_lines(warped_image, lane_lines(frame, lines))
	cv.imshow('Processed Image', result)
	test = average_lane_lines(lines)
	for line in test:
		if lines is None:
			continue
		if np.isnan(line).any():
			continue
		x1,y1,x2,y2 = line[0]
		hough_img = cv.line(warped_image, (int(x1),int(y1)),(int(x2),int(y2)),(0, 0, 255),5)
		cv.imshow("Hough Transformation",hough_img)

	if cv.waitKey(100) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()