"""
Aim: 
To compare distorted and undistorted pixel coordinates and compute distortion correction in pixels.

Summary:

1. Load a sample 3D point in world coordinates.
2. Extract intrinsics (focal length, principal point, image size) and extrinsics (camera pose) from Metashape.
3. Form the projection matrix (K × [R|t]) and manually project the 3D point to get undistorted 2D coordinates.
4. Use Metashape’s camera.project() to get distorted pixel coordinates.
5. Iteratively undistort the distorted coordinates using radial and tangential distortion coefficients.
6. Compare both undistorted results and output the pixel-level distortion correction.

"""


import Metashape
import numpy as np
import math

doc = Metashape.app.document
chunk = doc.chunk
camera = chunk.cameras[0]
calib = camera.sensor.calibration
crs = chunk.crs

T = chunk.transform.matrix

#Get a sample point in world coordinate system
points = chunk.point_cloud.points
point_3d = points[0].coord

#if scaling is performed, input the T.mulp(point_3d[:-1]) for unproject() i.e input the world coordinates , otherwise input point_3d[:-1]
point_3d_world = T.mulp(point_3d[:-1])

point_unproj = crs.unproject(point_3d_world)
point_internal = T.inv().mulp(point_unproj)
point_np = np.array([point_internal.x, point_internal.y, point_internal.z, 1])

f = calib.f
cx = calib.cx
cy = calib.cy
width = camera.sensor.width
height = camera.sensor.height
b1 = getattr(calib, 'b1', 0)  
b2 = getattr(calib, 'b2', 0)  
p3 = getattr(calib, 'p3', 0)  
p4 = getattr(calib, 'p4', 0)  

K = np.array([
    [f, 0, cx + width / 2],
    [0, f, cy + height / 2],
    [0, 0, 1]
])

RT_4x4 = np.array([[camera.transform.inv().row(i)[j] for j in range(4)] for i in range(4)])
RT = RT_4x4[:3, :]



proj_homog = K @ (RT @ point_np)
print(f"Projection matrix:{proj_homog}\n")
u_undistorted_manual = proj_homog[0] / proj_homog[2]
v_undistorted_manual = proj_homog[1] / proj_homog[2]
print(f"Manually calculated undistorted pixel coordinates: ({u_undistorted_manual:.2f}, {v_undistorted_manual:.2f})")

metashape_distorted = camera.project(point_internal)
print(f"Metashape calculated distorted pixel coordinates using .project(): ({metashape_distorted.x:.2f}, {metashape_distorted.y:.2f})")


# Distortion Section

x_dist = (metashape_distorted.x - (width/2 + cx)) / f
y_dist = (metashape_distorted.y - (height/2 + cy)) / f

print(f"Distorted coordinates: ({x_dist:.2f}, {y_dist:.2f})")


x_undist = x_dist
y_undist = y_dist

print("\nIterative undistortion from Metashape's distorted coordinates:")
for i in range(5):
    r_undist = math.sqrt(x_undist**2 + y_undist**2)
    x_rad = x_undist * (1 + calib.k1*r_undist**2 + calib.k2*r_undist**4 + calib.k3*r_undist**6 + calib.k4*r_undist**8)
    y_rad = y_undist * (1 + calib.k1*r_undist**2 + calib.k2*r_undist**4 + calib.k3*r_undist**6 + calib.k4*r_undist**8)
    x_tang = (calib.p1*(r_undist**2 + 2*x_undist**2) + 2*calib.p2*x_undist*y_undist) * (1 + p3*r_undist**2 + p4*r_undist**4)
    y_tang = (calib.p2*(r_undist**2 + 2*y_undist**2) + 2*calib.p1*x_undist*y_undist) * (1 + p3*r_undist**2 + p4*r_undist**4)
    dx = (x_rad + x_tang) - x_dist
    dy = (y_rad + y_tang) - y_dist
    x_undist = x_undist - dx
    y_undist = y_undist - dy
    print(f"Iteration {i+1}: x_undist={x_undist:.6f}, y_undist={y_undist:.6f}, error=({dx:.6f}, {dy:.6f})")

u_undistorted_from_distorted = width/2 + cx + x_undist*f + x_undist*b1 + y_undist*b2
v_undistorted_from_distorted = height/2 + cy + y_undist*f

print("\nResults comparison:")
print(f"Metashape distorted: ({metashape_distorted.x:.2f}, {metashape_distorted.y:.2f})")
print(f"Undistorted from Metashape's distorted: ({u_undistorted_from_distorted:.2f}, {v_undistorted_from_distorted:.2f})\n")

print(f"distortion: {metashape_distorted.x - u_undistorted_from_distorted:.2f}, {metashape_distorted.y - v_undistorted_from_distorted:.2f}\n")

print(f"Manual undistorted calculation: ({u_undistorted_manual:.2f}, {v_undistorted_manual:.2f})")
print(f"Difference: ({u_undistorted_manual-u_undistorted_from_distorted:.2f}, {v_undistorted_manual-v_undistorted_from_distorted:.2f})")
