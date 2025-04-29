"""
Aim:
To compute and visualize distortion plot.

Summary:
1. Load camera intrinsics and distortion coefficients from calibration.
2. Generate a grid of 2D pixel coordinates across the distorted image.
3. Undistort the pixel coordinates using OpenCV's `undistortPoints()`, then reproject them back using `projectPoints()`.
4. Compute displacement vectors between original (distorted) and reprojected (ideal) pixel locations.
5. Visualize distortion using quiver plot and report the maximum deviation in pixels.

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image_path') 

magnification = 10 

# intrinsic matrix (K) extracted from metashape calibration of 2m shot images
K = np.array([[8.88577658e+03, 0.00000000e+00, 2.76909487e+03],
    [0.00000000e+00, 8.88577658e+03, 2.06026385e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    , dtype=np.float32)

# distortion coefficients extracted from metashape calibration of 2m shot images
dist_coeffs = np.array([
    -7.41900169e-02, 4.65760013e-01, -1.32860699e-04, 1.95260138e-04, -9.99559360e-01
], dtype=np.float32)

# Get image dimensions
width, height = image.shape[1], image.shape[0]
print(f"Image size: {width} x {height}")


step = 100


x_vals, y_vals = np.meshgrid(
    np.arange(0, width, step),
    np.arange(0, height, step)
)


points_2D = np.stack([x_vals.ravel(), y_vals.ravel()], axis=-1).astype(np.float32)
normalized = cv2.undistortPoints(points_2D.reshape(-1, 1, 2), K, None)


reprojected, _ = cv2.projectPoints(
    np.concatenate([normalized, np.ones_like(normalized[:, :, :1])], axis=2),  # add z=1
    rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
    cameraMatrix=K, distCoeffs=dist_coeffs
)


points_distorted = reprojected.reshape(-1, 2)


diff = points_distorted - points_2D
errors = np.linalg.norm(diff, axis=1)
max_error = np.max(errors)
max_idx = np.argmax(errors)

# Report
print(f"Max distortion deviation: {max_error:.4f} pixels at point index {max_idx}")
print(f"Original:  {points_2D[max_idx]}")
print(f"Distorted: {points_distorted[max_idx]}")
print(f"Deviation vector: {diff[max_idx]}")


plt.figure(figsize=(10, 8))
plt.quiver(
    points_2D[:, 0], points_2D[:, 1],
    diff[:, 0]*magnification, diff[:, 1]*magnification,
    angles='xy', scale_units='xy', scale=1, color='red'
)
plt.gca().invert_yaxis()
plt.title("Distortion Displacement Vectors")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.show()


