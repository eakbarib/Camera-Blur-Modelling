import os
import cv2
import numpy as np
import json
import rawpy

# Checkerboard pattern: adjust if different
CHECKERBOARD = (9, 8)  # (columns, rows) of internal corners
SQUARE_SIZE = 1.0  # arbitrary units, since we care about relative intrinsics

def calibrate_camera(images):
    """
    Calibrate camera using checkerboard images.
    Returns camera matrix, distortion coefficients, or None if failed.
    """
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Checkerboard not found in image")

    if len(objpoints) < 5:
        print(f"Not enough valid images for calibration (need at least 5, got {len(objpoints)})")
        return None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        return mtx, dist
    else:
        print("Calibration failed")
        return None, None

if __name__ == "__main__":
    depth_groups_dir = "depth_groups"

    if not os.path.exists(depth_groups_dir):
        print(f"Directory {depth_groups_dir} does not exist. Run group_checkerboard_by_depth.py first.")
        exit(1)

    for depth_folder in sorted(os.listdir(depth_groups_dir)):
        depth_path = os.path.join(depth_groups_dir, depth_folder)
        if not os.path.isdir(depth_path) or not depth_folder.startswith("depth_"):
            continue

        print(f"Processing {depth_folder}...")

        # Load images
        images = []
        for filename in sorted(os.listdir(depth_path)):
            if filename.endswith(".CR3"):
                filepath = os.path.join(depth_path, filename)
                try:
                    img = rawpy.imread(filepath).postprocess()
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

        if not images:
            print(f"No images found in {depth_folder}")
            continue

        # Calibrate
        mtx, dist = calibrate_camera(images)

        if mtx is not None and dist is not None:
            # Save to JSON
            calib_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist()
            }
            json_path = os.path.join(depth_path, "calib.json")
            with open(json_path, 'w') as f:
                json.dump(calib_data, f, indent=4)
            print(f"Saved calibration to {json_path}")
        else:
            print(f"Calibration failed for {depth_folder}")

    print("Calibration complete.")