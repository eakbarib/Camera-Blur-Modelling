from pathlib import Path
import json
import numpy as np
import sqlite3
from PIL import Image
import os
import matplotlib.pyplot as plt
import pycolmap

# intrinsics have elements in the form [fx, fy, cx, cy, k1, k2, p1, p2]
# given a set of images and intrinsics, reconstruct 3d points and calculate reprojection errors
def reconstruct_3d(database_name, output_dir, image_dir, image_paths, intrinsics):    
    database_path = output_dir / database_name
    
    # create db if needed + get features
    os.system(f"colmap feature_extractor --image_path {str(image_dir)} --database_path {str(database_path)}"
              " --ImageReader.single_camera_per_image 1")
    
    # replace cameras infered from metadata with cameras from intrinsics
    
    cam_type = pycolmap.CameraModelId.OPENCV.value
    image_paths = [str(f) for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg']
    
    with sqlite3.connect(database_path) as db:
        cur = db.cursor()
        
        for i, params in enumerate(intrinsics):
            img_path = image_paths[i]
            filename = os.path.basename(img_path)
            
            # find camera of image
            cur.execute("SELECT camera_id FROM images WHERE name = ?", (filename,))
            cam_id = cur.fetchone()[0]
            
            # log old params
            cur.execute("SELECT params FROM cameras WHERE camera_id = ?", (cam_id,))
            old_params = np.frombuffer(cur.fetchone()[0], dtype=np.float64)
            print(old_params)
            print(params)
            
            # update camera
            cur.execute(
                "UPDATE cameras SET model=?, params=?, prior_focal_length=? WHERE camera_id=?", 
                (cam_type, params.tobytes(), 1, cam_id)
            )
            
        db.commit()
        
    # match images
    os.system(f"colmap exhaustive_matcher --database_path {str(database_path)}")
    
    # do sparse reconstruction
    os.system(f"colmap mapper --image_path {str(image_dir)} --database_path {str(database_path)} --output_path {str(output_dir)}"
               " --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0")
    
    reconstruction = pycolmap.Reconstruction(str(output_dir / '1'))
    print(reconstruction.summary())
    for camera_id, camera in reconstruction.cameras.items():
        print(camera_id, camera)
    #np.array([elm[1].xyz for elm in reconstruction.points3D.items()])

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

print(intrinsics)

reconstruct_3d("base_recon.db", output_dir, image_folder, image_paths, [intrinsics]*len(image_paths))