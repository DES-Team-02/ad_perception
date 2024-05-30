import numpy as np
import cv2 as cv

image = cv.imread('/home/niklas/Pictures/Screenshots/Screenshot from 2024-05-17 15-42-30.png')


cv.imshow('image', image)

cv.waitKey(0)
cv.destroyAllWindows()