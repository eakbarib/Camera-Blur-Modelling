from pathlib import Path
import json
import numpy as np
import sqlite3
from PIL import Image
import os
import pycolmap # this library is utter dogshit

# intrinsics have elements in the form [fx, fy, cx, cy, k1, k2, p1, p2]
# why do you need the folder and the image paths? colmap
# given a set of images and intrinsics, reconstruct 3d points and calculate reprojection errors
def reconstruct_3d(database_name, output_dir, image_dir, image_paths, intrinsics):
    # this function is absolutely disgusting
    
    database_path = output_dir / database_name
    
    # creates db if needed + gets features
    pycolmap.extract_features(database_path, image_dir) # your ram will not appreciate this
    
    # manually set cameras
    
    cam_type = pycolmap.CameraModelId.OPENCV.value
    image_paths = [str(f) for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg']
    
    with sqlite3.connect(database_path) as db:
        cur = db.cursor()
        cur.execute("DELETE FROM cameras;")
        for i, params in enumerate(intrinsics):
            img_path = image_paths[i]
            # lazy load with PIL so I can just get the metadata
            with Image.open(img_path) as img:
                width, height = img.size
            
            # add camera
            cur.execute(
                "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)"
                "VALUES (?, ?, ?, ?, ?, ?)", 
                (i+1, cam_type, width, height, params.tobytes(), 1)
            )
            
            # link corresponding image to camera
            cur.execute(
                "UPDATE images SET camera_id = ? WHERE name = ?", 
                (i+1, os.path.basename(img_path))
            )
            
        cur.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0;")
        print(f"Number of matched image pairs: {cur.fetchone()[0]}")
            
        db.commit()
    
    # match images (maybe this can be done before setting cams, but I'm not sure)
    pycolmap.match_exhaustive(database_path)
    
    # why do we have to specify image_dir again, all the paths are already in the db
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
    maps[0].write(output_dir)
    
    #print(reconstruction)
    
    
    return
    #return (points, errors) # todo
    
output_dir = Path("colmap/")

image_folder = Path('COLMAP_ROOM/')
image_paths = [str(f) for f in image_folder.iterdir() if f.is_file() and f.suffix.lower() == '.jpg']

# load camera parameters
with open('single_focus_checkerboard/calib.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    mtx = np.array(data["camera_matrix"])
    dist = data["distortion_coefficients"][0]
    intrinsics = np.array([mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2], dist[0], dist[1], dist[2], dist[3]], dtype=np.float64)

reconstruct_3d("base_recon.db", output_dir, image_folder, image_paths, [intrinsics]*len(image_paths))