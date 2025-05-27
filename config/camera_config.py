'''
This module provides a utility function to initialize and configure Raspberry Pi cameras using the Picamera2 API. It sets the resolution,
format, exposure time, and analogue gain.
'''

from picamera2 import Picamera2

def setup_camera(index, size=(416, 416), exposure=4000, gain=1.0):
    cam = Picamera2(index)
    cam.preview_configuration.main.size = size
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
    cam.start()
    return cam