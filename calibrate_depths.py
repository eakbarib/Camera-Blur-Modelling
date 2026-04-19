import os
import cv2
import numpy as np
import json

# Checkerboard pattern: adjust if different
CHECKERBOARD = (9, 8)  # (columns, rows) of internal corners
SQUARE_SIZE = 1.0  # arbitrary units, since we care about relative intrinsics
OBJ_POINT_TEMPLATE = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
OBJ_POINT_TEMPLATE[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE


def get_sorted_image_paths(folder):
    """
    Return sorted image paths using numeric filename ordering.
    """
    def numeric_key(name):
        base = os.path.splitext(name)[0]
        try:
            return int(base)
        except ValueError:
            return base

    filenames = [entry.name for entry in os.scandir(folder) if entry.is_file() and entry.name.lower().endswith('.jpg')]
    filenames.sort(key=numeric_key)
    return [os.path.join(folder, name) for name in filenames]


def get_checkerboard_points(image_paths, verbose=False):
    """
    Extract checkerboard corner points from images.
    Returns objpoints (3D), imgpoints (2D), and image shape.
    """
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    img_shape = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in image_paths:
        img = None
        if verbose:
            print(f"Reading image: {img_path}")
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Unable to load image: {img_path}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)
        if not found:
            if verbose:
                print(f"Checkerboard not found in image: {img_path}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(OBJ_POINT_TEMPLATE)
        imgpoints.append(corners)

    return objpoints, imgpoints, img_shape

def get_sorted_image_paths(folder):
    """
    Return sorted image paths using numeric filename ordering.
    """
    def numeric_key(filename):
        name = os.path.splitext(filename)[0]
        try:
            return int(name)
        except ValueError:
            return filename

    filenames = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    filenames.sort(key=numeric_key)
    return [os.path.join(folder, f) for f in filenames]

def calibrate_camera(image_paths):
    """
    Calibrate camera using checkerboard images.
    Returns reprojection error, camera matrix, distortion coefficients, rotation vectors, translation vectors, or None if failed.
    """
    objpoints, imgpoints, img_shape = get_checkerboard_points(image_paths)

    if len(objpoints) < 5:
        print(f"Not enough valid images for calibration (need at least 5, got {len(objpoints)})")
        return None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    if ret:
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("Calibration failed")
        return None, None, None, None, None

def load_or_calibrate_initial(single_focus_dir):
    """
    Load or calibrate the initial camera parameters from Single Focus Calibration images.
    Returns (mtx, dist, reprojection_error).
    """
    calib_json = os.path.join(single_focus_dir, "calib.json")
    
    # Try to load existing calibration
    if os.path.exists(calib_json):
        try:
            with open(calib_json, 'r') as f:
                data = json.load(f)
            mtx = np.array(data["camera_matrix"], dtype=np.float32)
            dist = np.array(data["distortion_coefficients"], dtype=np.float32)
            reprojection_error = float(data.get("reprojection_error", 0.0))
            print(f"Loaded initial calibration from {calib_json}")
            return mtx, dist, reprojection_error
        except Exception as e:
            print(f"Error loading calibration: {e}")
    
    # Otherwise, calibrate from images
    print("Calibrating initial camera parameters from Single Focus Calibration...")
    image_paths = get_sorted_image_paths(single_focus_dir)
    
    if not image_paths:
        print(f"No images found in {single_focus_dir}")
        return None, None, None
    
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(image_paths)
    
    if ret is not None and mtx is not None and dist is not None:
        # Save for future use
        calib_data = {
            "reprojection_error": float(ret),
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist()
        }
        with open(calib_json, 'w') as f:
            json.dump(calib_data, f, indent=4)
        print(f"Saved initial calibration to {calib_json}")
        return mtx, dist, ret
    else:
        print("Initial calibration failed")
        return None, None, None

def is_valid_calib_json(json_path):
    """
    Determine whether an existing calib.json contains valid calibration data.
    """
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    if "camera_matrix" not in data or "distortion_coefficients" not in data:
        return False
    
    return True

def calibrate_depth_camera(objpoints, imgpoints, img_shape, initial_mtx, initial_dist):
    """
    Calibrate camera matrix for a specific depth using initial calibration as initialization.
    Uses cv2.CALIB_USE_INTRINSIC_GUESS to refine the matrix.
    Returns reprojection error, refined camera matrix, distortion coefficients, rotation vectors, and translation vectors, or None if failed.
    """
    if len(objpoints) < 5:
        print(f"Not enough valid images for calibration (need at least 5, got {len(objpoints)})")
        return None, None, None, None, None
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        initial_mtx.copy(),
        initial_dist.copy(),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    
    if ret:
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("Calibration refinement failed")
        return None, None, None, None, None

def compute_mean_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    Compute mean reprojection error from calibration outputs.
    Uses returned rotation and translation vectors instead of re-solving PnP.
    """
    if len(objpoints) == 0 or len(objpoints) != len(rvecs):
        return None, 0

    total_error = 0.0
    for obj_pts, img_pts, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
        total_error += cv2.norm(img_pts, projected_pts, cv2.NORM_L2) / len(img_pts)

    mean_error = total_error / len(objpoints)
    return mean_error, len(objpoints)

if __name__ == "__main__":
    depth_groups_dir = "depth_groups"
    single_focus_dir = "single_focus_checkerboard"
    
    skip_complete = True # skips folders with an existing calibration

    if not os.path.exists(depth_groups_dir):
        print(f"Directory {depth_groups_dir} does not exist. Run group_checkerboard_by_depth.py first.")
        exit(1)

    if not os.path.exists(single_focus_dir):
        print(f"Directory {single_focus_dir} does not exist.")
        exit(1)

    # Load or calibrate initial camera parameters
    print("=" * 60)
    print("Loading/Calibrating initial camera parameters...")
    print("=" * 60)
    initial_mtx, initial_dist, initial_reprojection_error = load_or_calibrate_initial(single_focus_dir)
    
    if initial_mtx is None or initial_dist is None:
        print("Failed to load or calibrate initial camera parameters. Exiting.")
        exit(1)
    
    print(f"Initial calibration - Reprojection Error: {initial_reprojection_error:.6f}")
    print("=" * 60)

    processed_count = 0
    skipped_count = 0

    for depth_folder in sorted(os.listdir(depth_groups_dir)):
        depth_path = os.path.join(depth_groups_dir, depth_folder)
        if not os.path.isdir(depth_path) or not depth_folder.startswith("depth_"):
            continue
        
        json_path = os.path.join(depth_path, "calib.json")
        if skip_complete and is_valid_calib_json(json_path):
            skipped_count += 1
            continue

        print(f"Processing {depth_folder}...")
        processed_count += 1

        # Load images
        image_paths = get_sorted_image_paths(depth_path)

        if not image_paths:
            print(f"No images found in {depth_folder}")
            continue

        objpoints, imgpoints, img_shape = get_checkerboard_points(image_paths)

        if len(objpoints) < 5:
            print(f"Not enough valid images for {depth_folder} (need at least 5, got {len(objpoints)})")
            continue

        # Calibrate camera matrix for this depth using initial calibration as initialization
        ret, depth_mtx, depth_dist, depth_rvecs, depth_tvecs = calibrate_depth_camera(objpoints, imgpoints, img_shape, initial_mtx, initial_dist)

        if ret is not None and depth_mtx is not None and depth_dist is not None:
            # Calculate reprojection error using the refined depth-specific calibration outputs
            mean_error, valid_count = compute_mean_reprojection_error(objpoints, imgpoints, depth_mtx, depth_dist, depth_rvecs, depth_tvecs)
            
            if mean_error is not None and valid_count > 0:
                # Save to JSON with depth-refined calibration and reprojection error
                calib_data = {
                    "initial_calibration_error": float(initial_reprojection_error),
                    "camera_matrix": depth_mtx.tolist(),
                    "distortion_coefficients": depth_dist.tolist(),
                    "calibration_error": float(ret),
                    "mean_reprojection_error": float(mean_error),
                    "valid_images_count": int(valid_count)
                }
                with open(json_path, 'w') as f:
                    json.dump(calib_data, f, indent=4)
                print(f"Saved calibration to {json_path}")
                print(f"  - Calibration error: {ret:.6f}")
                print(f"  - Mean reprojection error: {mean_error:.6f}, valid images: {valid_count}")
            else:
                print(f"Could not calculate reprojection error for {depth_folder}")
                calib_data = {"error": "Could not calculate reprojection error"}
                with open(json_path, 'w') as f:
                    json.dump(calib_data, f, indent=4)
        else:
            print(f"Calibration refinement failed for {depth_folder}")
            calib_data = {"error": "Calibration refinement failed"}
            with open(json_path, 'w') as f:
                json.dump(calib_data, f, indent=4)

    print(f"Calibration complete. Processed: {processed_count}, Skipped: {skipped_count}")