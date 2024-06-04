from pydbus import SessionBus
import pydbus_module

from lane_follower import LaneFollower
from image_processor import *

import cv2 as cv
import time

from jetcam.csi_camera import CSICamera


# PyDbus Setting
bus = SessionBus()
service = pydbus_module.VehicleControlDBusService()
bus.publish("com.team2.VehicleControl", service)
print("VehicleControl D-Bus service is running.")

# Controller Init
lane_follower = LaneFollower(width=1280, height=720, max_steer=1.0, normal_throttle=1.0)

# # Image Init
# cap = cv.VideoCapture('camera_test/output.avi')
camera = CSICamera(width=1280, height=720, capture_width=1280, capture_height=720, capture_fps=5)

# while cap.isOpened():
#     ret, frame = cap.read()
while 1:
    image = camera.read()
    #cv.imshow('frame', frame)
    start_time = time.time()
    # middle_points = frame_processor(frame)
    middle_points = frame_processor(image)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    lane_follower.calculate_control(middle_points)
    throttle, steer = lane_follower.get_control()
    
    service.SetThrottle(throttle * 0.2)
    service.SetSteering(steer * -1.0)

    if cv.waitKey(1) == ord('q'):
        break

# cap.release()
cv.destroyAllWindows()
