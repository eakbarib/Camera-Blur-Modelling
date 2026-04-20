import os
import cv2
import numpy as np
import json
from common import *
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

CHECKERBOARD = (9, 8)  # (columns, rows) of internal corners
SQUARE_SIZE = 1.0  # arbitrary units, since we care about relative intrinsics
OBJ_POINT_TEMPLATE = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
OBJ_POINT_TEMPLATE[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

def get_checkerboard_points_single(img_path, criteria, downscale=4, verbose=False):
    """
    Subprocess of get_checkerboard_points
    gets the checkerboard points for a single image
    """
    gray = None
    try:
        if verbose:
            print(f"Reading image: {img_path}")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Unable to load image: {img_path}")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return

    gray_small = cv2.resize(gray, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_AREA)

    found, corners = cv2.findChessboardCorners(gray_small, CHECKERBOARD, flags=cv2.CALIB_CB_FAST_CHECK)
    if not found:
        if verbose:
            print(f"Checkerboard not found in image: {img_path}")
        return

    corners = cv2.cornerSubPix(gray, corners*downscale, (11, 11), (-1, -1), criteria)
    return (OBJ_POINT_TEMPLATE, corners)

def get_checkerboard_points(image_paths, verbose=False):
    """
    Extract checkerboard corner points from images.
    Returns objpoints (3D), imgpoints (2D), and image shape.
    """
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # corner detection is very slow, so multiprocessing is needed
    max_workers = max(os.cpu_count() - 1, 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_checkerboard_points_single, path, criteria, downscale=4, verbose=verbose) 
            for path in image_paths
        ]
        
        for future in futures:
            result = future.result()
            if result:
                o_pt, i_pt = result
                objpoints.append(o_pt)
                imgpoints.append(i_pt)
                
    # lazy load shape metadata
    with Image.open(image_paths[0]) as img:
        img_shape = img.size

    return objpoints, imgpoints, img_shape

def calibrate_camera(image_paths, initialize=None, doLU=False):
    """
    Calibrate camera using checkerboard images.
    Set initialize to (camera_matrix, distortion_coefficents) to refine from an initial estimate
    Returns reprojection error, camera matrix, distortion coefficients, rotation vectors, translation vectors, or None if failed.
    """
    objpoints, imgpoints, img_shape = get_checkerboard_points(image_paths)

    if len(objpoints) < 5:
        print(f"Not enough valid images for calibration (need at least 5, got {len(objpoints)})")
        return None, None, None, None, None
    
    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    if initialize is None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, 
            imgpoints, 
            img_shape,
            None,
            None,
            flags=flags | (cv2.CALIB_USE_LU if doLU else 0)
        )
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            img_shape,
            initialize[0].copy(),
            initialize[1].copy(),
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_USE_LU if doLU else 0)
        )
    if ret:
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("Calibration failed")
        return None, None, None, None, None

def calibration_as_json(ret, mtx, dist):
    """Returns the json convertible equivalent of a calibration"""
    
    if ret is not None and mtx is not None and dist is not None:
        return {
            "reprojection_error": float(ret),
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist()
        }
    else:
        return {}

def calibrate_depths(skip_complete=True, calibrate_stacks=False):

    # Load or calibrate initial camera parameters (from checkerboard singles)
    
    single_calib_path = checkerboard_single_path / "calib.json"
    # load existing
    if skip_complete and single_calib_path.exists():
        with single_calib_path.open(mode='r') as f:
            calib = json.load(f)
        initial_reprojection_error = calib["reprojection_error"]
        initial_mtx = np.array(calib["camera_matrix"])
        initial_dist = np.array(calib["distortion_coefficients"])
    
    # calibrate checkerboard singles
    else:
        print(f"Calibrating checkerboard singles")
        single_image_paths = [f for f in checkerboard_single_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']]
        
        initial_reprojection_error, initial_mtx, initial_dist, _, _ = calibrate_camera(single_image_paths, doLU=True)
        
        # save result
        calib_data = calibration_as_json(initial_reprojection_error, initial_mtx, initial_dist)
        with single_calib_path.open(mode='w') as f:
            json.dump(calib_data, f, indent=4)
    
    # early out if intial calibration somehow fails
    if initial_mtx is None or initial_dist is None:
        print("Failed to load or calibrate initial camera parameters. Exiting.")
        exit(1)
    
    print(f"Initial calibration - Reprojection Error: {initial_reprojection_error:.6f}")
    print("=" * 60)
    
    if not calibrate_stacks:
        exit(0)

    processed_count = 0
    skipped_count = 0
    
    # initialize output
    calib_file_path = checkerboard_stack_path / "calibs.json"
    if calib_file_path.exists():
        with calib_file_path.open(mode='r') as file:
            calibs = json.load(file)
    else:
        with calib_file_path.open(mode='w') as file:
            json.dump({}, file, indent=4)
            
    # lazy load depth groups
    from depthGroup import depth_groups

    # calibrate each depth group
    for group in depth_groups:
        if skip_complete and group.read_calibration() is not None:
            skipped_count += 1
            continue

        print(f"Processing depth {group.depth}...")
        processed_count += 1

        if not group.image_paths or len(group.image_paths) == 0:
            print(f"No images found in depth {group.depth}")
            continue

        # Calibrate camera matrix for this depth using initial calibration as initialization
        ret, depth_mtx, depth_dist, _, _ = calibrate_camera(group.image_paths, initialize=(initial_mtx, initial_dist))

        # save to file
        calib_data = calibration_as_json(ret, depth_mtx, depth_dist)
        calibs[str(group.depth)] = calib_data
        with open(calib_file_path, 'w') as f:
            json.dump(calibs, f, indent=4)
        
        if calib_data == {}:
            print(f"Calibration refinement failed for depth {group.depth}")
        else:
            print(f"  - Calibration error: {ret:.6f}")

    print(f"Calibration complete. Processed: {processed_count}, Skipped: {skipped_count}")