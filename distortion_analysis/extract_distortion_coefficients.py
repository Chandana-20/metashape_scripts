"""
Aim:
To extract the camera intrinsic matrix and distortion coefficients from Metashape calibration data.

"""
import Metashape
doc = Metashape.app.document
import numpy as np

chunk = doc.chunk
camera = chunk.cameras[0]
calib = camera.sensor.calibration

f = calib.f
cx = calib.cx
cy = calib.cy
width = camera.sensor.width
height = camera.sensor.height

mtx = np.array([
    [f, 0, cx + width / 2],
    [0, f, cy + height / 2],
    [0, 0, 1]
])

print(f"Intrinsic matrix:\n{mtx}\n")
dist = np.array([calib.k1, calib.k2, calib.p1, calib.p2, calib.k3]) 
print(f"dist: {dist}\n")


