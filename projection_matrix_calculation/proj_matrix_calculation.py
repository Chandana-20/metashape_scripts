"""
AIM: To manually reconstruct the projection matrix (K [Intrinsic] × [R|t] [Extrinsic]) from Metashape calibration data 
and validate it by projecting a known 3D world coordinate onto the image plane.


Summary:

1. Get Intrinsics (from Metashape sensor calibration: focal length, principal point, image size).
2. Get Extrinsics (camera pose matrix: rotation and translation from camera transform).
3  Form the full Projection matrix manually.
4. Project a world 3D point into 2D pixel space.
5. Cross-check the result with Metashape's own projection. 

"""

import Metashape
import numpy as np

doc = Metashape.app.document
chunk = doc.chunk
camera = chunk.cameras[0]
crs = chunk.crs

T = chunk.transform.matrix

#Get a sample point in world coordinate system
points = chunk.point_cloud.points
point_3d = points[0].coord

#if scaling is performed, input the T.mulp(point_3d[:-1]) for unproject() i.e input the world coordinates , otherwise input point_3d[:-1]
point_3d_world = T.mulp(point_3d[:-1])

point_unproj = crs.unproject(point_3d_world)
print(f"Unprojected point: {point_unproj}")
point_internal = T.inv().mulp(point_unproj)
point_np = np.array([point_internal.x, point_internal.y, point_internal.z, 1])

# calculation of Intrinsic matrix
calib = camera.sensor.calibration
f = calib.f
cx = calib.cx
cy = calib.cy
width = camera.sensor.width
height = camera.sensor.height

K = np.array([
    [f, 0, cx + width / 2],
    [0, f, cy + height / 2],
    [0, 0, 1]
])


#calculation of Extrinsic matrix
# Convert Metashape Matrix to NumPy array (4x4)
RT_4x4 = np.array([[camera.transform.inv().row(i)[j] for j in range(4)] for i in range(4)])
RT = RT_4x4[:3, :]  # Extract 3x4 matrix for projection

print("Projection Matrix K * RT:")
print(K @ RT)

# Projection
proj_homog = K @ (RT @ point_np)

u = proj_homog[0] / proj_homog[2]
v = proj_homog[1] / proj_homog[2]

print(f"3D point in world coordinates: {point_3d_world}")
print(f"Projected 2D pixel coordinates: ({u:.2f}, {v:.2f})")
print(f"metashape 2D coordinates: {camera.project(point_internal)}")





