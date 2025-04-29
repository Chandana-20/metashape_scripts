"""
Aim:
To export the pixel coordinates (u, v) of tie points visible in at least two camera images from a Metashape project.

Summary:
1. Set up an output CSV file for saving the tie point data.
2. Build a mapping from track IDs to point indices in the dense point cloud.
3. Count how many cameras observe each tie point.
4. For each tie point observed in at least two cameras, record its pixel coordinates (u, v) along with the camera label.
5. Save the collected tie point data into a CSV file for further analysis.


"""

import csv
import Metashape
import os

# Setup output directory 
output_dir = r"D:\Chandana\ArUco_Accuracy\Scripts\tiepoints_visualisation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "tiepoints_uv_TwoMeters.csv")

# Open file
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Point_ID", "Camera", "u", "v"])

    # Get chunk
    chunk = Metashape.app.document.chunk

    # Build a lookup: track_id -> point index
    track_id_to_point_index = {
        point.track_id: i
        for i, point in enumerate(chunk.point_cloud.points)
    }

    # Count cameras per point
    camera_count = {}
    for camera in chunk.cameras:
        if not camera.transform:
            continue
        for proj in chunk.point_cloud.projections[camera]:
            point_index = track_id_to_point_index.get(proj.track_id)
            if point_index is not None:
                camera_count[point_index] = camera_count.get(point_index, 0) + 1

    # Write tie points with at least 2 camera projections
    for camera in chunk.cameras:
        if not camera.transform:
            continue
        for proj in chunk.point_cloud.projections[camera]:
            point_index = track_id_to_point_index.get(proj.track_id)
            if point_index is not None and camera_count.get(point_index, 0) >= 2:
                u, v = proj.coord
                writer.writerow([point_index, camera.label, u, v])

print(f"Tie points saved to: {output_path}")
