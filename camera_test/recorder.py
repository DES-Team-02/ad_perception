from jetcam.csi_camera import CSICamera

import numpy as np
import cv2 as cv

camera = CSICamera(width=1280, height=720, capture_width=1920, capture_height=1080, capture_fps=30)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('new_output.avi', fourcc, 15, (1280, 720))
 
while True:
    image = camera.read()
    out.write(image)

    # cv.imshow('image', image)

    if cv.waitKey(1) == ord('q'):
        break
    
out.release()
cv.destroyAllWindows()
