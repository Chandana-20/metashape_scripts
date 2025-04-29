"""
Aim : To check if the selected tie point has its projection in the selcted camera.

Summary : 
1. Extracts a specific point from the point cloud and its corresponding track ID.
2. Retrieves the camera from the chunk.
3. Checks if the specified point is observed in this camera by querying the projections collection.

"""

import Metashape

# Access the active document and chunk
doc = Metashape.app.document
chunk = doc.chunk


# Get the first point and its track ID
point = chunk.point_cloud.points[1]
track_id = point.track_id
point_3d = point.coord 

# Get the first camera
camera = chunk.cameras[0]

# Check if this point is observed in the camera using the correct API
is_observed = False
proj = chunk.point_cloud.projections[camera][track_id]
if proj:
    print(f"Point {track_id} is observed in camera {camera.label} at pixel coordinates {proj.coord}")
    print(f"The 3D coordinates of this point are: {point_3d}")
    is_observed = True
else:
    print(f"Point {track_id} is NOT observed in camera {camera.label}")