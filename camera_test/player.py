import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture('new_output.avi')
 
while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    cv.imshow('frame', frame)
    cv.imshow('gray', gray)

    if cv.waitKey(1000) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()