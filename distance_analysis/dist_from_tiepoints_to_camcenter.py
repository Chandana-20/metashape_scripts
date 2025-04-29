"""
Aim : To calculate the distances between the camera center and each tiepoint which is projected on that camera.

Summary
1. Accesses the first camera (cam0) and its point cloud, then iterates through all projections to find valid tiepoints. 
2. For each valid tiepoint, it calculates the Euclidean distance from the camera center, scales it using the chunk transformation matrix and
   stores distances and  as an array. 

Saves results in three files:

distances_TwoMeters.npy: NumPy array of all calculated distances
tiepoints_TwoMeters.npy: NumPy array containing 3D coordinates of all tiepoints
tiepoint_distances_TwoMeters.json: JSON dictionary mapping point indices to their distances


"""

import Metashape
import numpy as np
import json
import csv

import os



doc = Metashape.app.document
chunk = doc.chunk
cam0 = chunk.cameras[0]
cam0_center = cam0.center

point_cloud = chunk.point_cloud

points = point_cloud.points
projections = point_cloud.projections
npoints = len(points)

point_index = 0

point_coords = []
distances = []

tiepoint_distances={}

for proj in projections[cam0]:
    track_id = proj.track_id
    while point_index < npoints and points[point_index].track_id < track_id:
        point_index += 1
    if point_index < npoints and points[point_index].track_id == track_id:
        if not points[point_index].valid:
            continue
        else:
            points[point_index].selected = True  
            p = points[point_index].coord
            xyz = np.array(p[:-1])
            point_coords.append(xyz)
            dist = (cam0_center - p[:-1]).norm() * chunk.transform.matrix.scale()
            distances.append(dist)
            # tiepoint_distances[tuple(xyz)] = dist
            tiepoint_distances[point_index] = dist 





project_path = doc.path  # full path to .psx file

if project_path:
    project_dir = os.path.dirname(project_path)
else:
    project_dir = os.getcwd() 

output_dir = os.path.join(project_dir, "output_dist_calculation")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "distances_TwoMeters.npy"), distances)
np.save(os.path.join(output_dir, "tiepoints_TwoMeters.npy"), np.array(point_coords))

with open(os.path.join(output_dir, "tiepoint_distances_TwoMeters.json"), "w") as f:
    json.dump(tiepoint_distances, f)

print(f"Saved files in: {os.path.abspath(output_dir)}")