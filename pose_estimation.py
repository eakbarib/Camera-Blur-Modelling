import os
import cv2
import numpy as np
import json
import rawpy
from imgSet import bokeh_img_sets
from analysis import get_depths

METERS_PER_PIXEL = 0.00010254

import json
import numpy as np

def generate_3d_from_json(detected_ids, json_path, meters_per_pixel):
    """
    Generates 3D physical coordinates directly from the layout JSON.
    """
    with open(json_path, 'r') as f:
        marker_data = json.load(f)
        
    # Create a quick lookup dictionary: {id: {"origin": [x,y], "size": s}}
    layout = {item["id"]: item for item in marker_data}
    
    obj_points = []
    
    for marker_id in detected_ids:
        idx = int(marker_id[0] if isinstance(marker_id, (list, np.ndarray)) else marker_id)
        
        if idx not in layout:
            continue
            
        # Get digital pixel values
        px_x, px_y = layout[idx]["origin"]
        px_size = layout[idx]["size"]
        
        # Convert pixels to physical meters!
        center_x = (px_x + (px_size / 2.0)) * meters_per_pixel
        center_y = (px_y + (px_size / 2.0)) * meters_per_pixel
        half_size = (px_size / 2.0) * meters_per_pixel
        center_z = 0.0
        
        # OpenCV Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        corners = [
            [center_x - half_size, center_y - half_size, center_z],
            [center_x + half_size, center_y - half_size, center_z],
            [center_x + half_size, center_y + half_size, center_z],
            [center_x - half_size, center_y + half_size, center_z]
        ]
        obj_points.append(corners)
        
    return np.array(obj_points, dtype=np.float32).reshape(-1, 3)

def find_calib_photo_corners(image):
    """
    Detects 5x5_50 ArUco markers in the calibration photograph and refines 
    their corners to sub-pixel accuracy.
    
    Args:
        image: A numpy array representing the photograph (BGR or Grayscale).
        
    Returns:
        corners: A list of 2D numpy arrays containing the 4 corners of each detected marker.
        ids: A 1D numpy array of the detected marker IDs.
    """
    # 1. Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Setup the exact dictionary from your generation script
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Modern OpenCV (4.7+) syntax
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 3. Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # 4. Error Handling
    if ids is None:
        print("Warning: No ArUco markers detected in this image.")
        return None, None

    # 5. Sub-pixel Refinement (Crucial for 3D accuracy)
    # We use a small 5x5 search window to avoid grabbing the white calibration dots by mistake
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    for i in range(len(corners)):
        cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)

    return corners, ids

def solve_pnp(objpoints, imgpoints, K, D):
    """Solve PnP and return 4x4 transformation matrix."""
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, K, D)
    if not ret:
        return None

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)

    # Form 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T

if __name__ == "__main__":
    depth_groups_dir = "depth_groups"

    results = {}

    for set_key, img_set in bokeh_img_sets.items():
        print(f"Processing imgset {set_key} ({img_set.id})...")

        # Get sharp image (in-focus)
        try:
            sharp_img = img_set.read_img(img_set.in_focus)
        except Exception as e:
            print(f"Error reading sharp image for {set_key}: {e}")
            continue

        # Compute depth
        fmin, fmax = get_depths(img_set)
        depth = (fmin[img_set.in_focus] + fmax[img_set.in_focus]) / 2.0
        depth_rounded = round(depth, 0)

        # Find closest available depth folder
        available_depths = []
        if os.path.exists(depth_groups_dir):
            for folder in os.listdir(depth_groups_dir):
                if folder.startswith("depth_"):
                    try:
                        d = float(folder.split("_")[1])
                        available_depths.append(d)
                    except ValueError:
                        continue

        if not available_depths:
            print(f"No depth folders found in {depth_groups_dir}")
            continue

        # Find closest depth
        closest_depth = min(available_depths, key=lambda x: abs(x - depth_rounded))
        depth_folder = f"depth_{closest_depth}"
        json_path = os.path.join(depth_groups_dir, depth_folder, "calib.json")

        if not os.path.exists(json_path):
            print(f"Calibration JSON not found for closest depth {closest_depth}: {json_path}")
            continue

        # Load K and D
        with open(json_path, 'r') as f:
            calib = json.load(f)
        K = np.array(calib["camera_matrix"])
        D = np.array(calib["distortion_coefficients"])

        # Undistort image
        undistorted = cv2.undistort(sharp_img, K, D)

        # Find ArUco marker corners
        corners, ids = find_calib_photo_corners(undistorted)
        if corners is None or ids is None:
            print(f"No ArUco markers found in undistorted image for {set_key}")
            continue

        # Generate 3D object points from JSON layout
        objpoints = generate_3d_from_json(ids, f"./calib_images/calib_{img_set.id}.json", METERS_PER_PIXEL)
        if objpoints.size == 0:
            print(f"No valid object points generated for {set_key}")
            continue

        # Flatten corners to match objpoints format
        imgpoints = np.array(corners).reshape(-1, 2).astype(np.float32)

        # Solve PnP
        T = solve_pnp(objpoints, imgpoints, K, D)
        if T is None:
            print(f"PnP failed for {set_key}")
            continue

        results[img_set.id] = {
            "depth": depth,
            "depth_rounded": depth_rounded,
            "transformation_matrix": T.tolist()
        }

        print(f"Success for {img_set.id}: depth {depth_rounded}, T shape {T.shape}")

    # Save results
    with open("pose_estimations.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Pose estimation complete. Results saved to pose_estimations.json")