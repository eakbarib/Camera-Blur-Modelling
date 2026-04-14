from pathlib import Path
import pycolmap as cm # this library is utter dogshit

# image shape is in numpy (h,w) form
# intrinsics have elements in the form [fx, fy, cx, cy, k1, k2, p1, p2]
# given a set of images and intrinsics, reconstruct 3d points and calculate reprojection errors
def reconstruct_3d(image_paths, image_shape, intrinsics):
    # this function is disgusting, I hate bash scripters
    
    database_path = Path("pycolmap/database.db")
    if database_path.exists():
        database_path.unlink()
    db = cm.Database(database_path)
    
    cameras = []
    for i, img_path in enumerate(image_paths):
        camera_id = db.add_camera(
            model="OPENCV",
            width=image_shape.shape[1],
            height=image_shape.shape[0],
            params=intrinsics[i]
        )
        cameras.push(camera_id)
        db.add_image(name=Path(img_path).name, camera_id=camera_id, image_id=i+1)

    db.commit()
    
    cm.extract_features(database_path, image_dir=Path("/"))
    cm.match_exhaustive(database_path)
    
    reconstruction = cm.incremental_mapping(
        database_path=database_path,
        image_path=Path("pycolmap/"),
        output_path=Path("pycolmap/reconstruction")
    )
    
    print(reconstruction)
    
    return
    #return (points, errors) # todo