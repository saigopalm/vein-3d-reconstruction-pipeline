'''
This code is for capturing checkerboard images on both the cameras used to calibrate the cameras.
The code captures images of the checkerboard pattern on both the cameras and saves them in separate folders.
'''

import cv2
import os
from picamera2 import Picamera2

# checkerboard dimensions (adjust according to your checkerboard)
CHECKERBOARD = (7, 6)

left_folder = "left"
right_folder = "right"
os.makedirs(left_folder, exist_ok=True)
os.makedirs(right_folder, exist_ok=True)

window_size = 416
picam1 = Picamera2(0)
picam1.preview_configuration.main.size = (window_size, window_size)
picam1.preview_configuration.main.format = "RGB888"
picam1.preview_configuration.align()
picam1.configure("preview")
picam1.set_controls({"ExposureTime": 4000, "AnalogueGain": 1.0})
picam1.start()

picam2 = Picamera2(1)
picam2.preview_configuration.main.size = (window_size, window_size)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"ExposureTime": 4000, "AnalogueGain": 1.0})
picam2.start()

image_counter = 0

def capture_images():
    global image_counter

    frame_left = picam1.capture_array()
    frame_right = picam2.capture_array()

    left_image_path = os.path.join(left_folder, f"im{image_counter + 1}.png")
    right_image_path = os.path.join(right_folder, f"im{image_counter + 1}.png")

    cv2.imwrite(left_image_path, frame_left)
    cv2.imwrite(right_image_path, frame_right)

    image_counter += 1
    print(f"Captured set {image_counter}: Left -> {left_image_path}, Right -> {right_image_path}")

while True:
    frame_left = picam1.capture_array()
    frame_right = picam2.capture_array()

    combined_frame = cv2.hconcat([frame_left, frame_right])

    cv2.putText(combined_frame, f"Captured Sets: {image_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Stereo Camera Calibration", combined_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Press 'c' to capture images
        capture_images()
    elif key == ord('q'):  # Press 'q' to quit
        break

picam1.stop()
picam2.stop()
cv2.destroyAllWindows()