from jetcam.csi_camera import CSICamera

import numpy as np
import cv2 as cv

camera = CSICamera(width=1280, height=720, capture_width=1920, capture_height=1080, capture_fps=30)

while True:
    image = camera.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    cv.imshow('image', image)
    cv.imshow('gray', gray)

    if cv.waitKey(1) == ord('q'):
        break
 
cv.destroyAllWindows()