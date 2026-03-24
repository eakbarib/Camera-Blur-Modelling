import os
import json
import numpy as np
from scipy import interpolate
import argparse

# Assumed sensor dimensions for Canon EOS R5 C (adjust if different camera)
SENSOR_WIDTH_MM = 36  # mm
SENSOR_HEIGHT_MM = 24  # mm

def load_calibrations(depth_groups_dir):
    """Load all calibration data from depth folders."""
    calibrations = {}
    for folder in os.listdir(depth_groups_dir):
        if folder.startswith("depth_"):
            try:
                depth = float(folder.split("_")[1])
                json_path = os.path.join(depth_groups_dir, folder, "calib.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        calib = json.load(f)
                    calibrations[depth] = {
                        'K': np.array(calib['camera_matrix']),
                        'D': np.array(calib['distortion_coefficients'])
                    }
            except (ValueError, KeyError):
                continue
    return calibrations

def interpolate_calibration(calibrations, target_depth):
    """Interpolate K and D matrices for target depth."""
    depths = sorted(calibrations.keys())
    if target_depth in depths:
        return calibrations[target_depth]

    if len(depths) < 2:
        raise ValueError("Need at least 2 calibration depths for interpolation")

    # Interpolate K matrix (3x3)
    K_values = np.array([calibrations[d]['K'].flatten() for d in depths])
    K_interp = interpolate.interp1d(depths, K_values, axis=0, kind='linear')
    K_flat = K_interp(target_depth)
    K = K_flat.reshape(3, 3)

    # Interpolate D coefficients
    D_values = np.array([calibrations[d]['D'].flatten() for d in depths])
    D_interp = interpolate.interp1d(depths, D_values, axis=0, kind='linear')
    D = D_interp(target_depth)

    return {'K': K, 'D': D}

def calculate_fov(K, sensor_width_mm, sensor_height_mm):
    """Calculate FOV in degrees from camera matrix."""
    fx = K[0, 0]
    fy = K[1, 1]

    # FOV = 2 * atan(sensor_size / (2 * focal_length_pixels))
    # But focal_length_pixels = fx, and sensor_size in mm, but need consistent units
    # Actually, FOV = 2 * atan(sensor_width_mm / (2 * fx * pixel_size_mm))
    # But we don't have pixel_size. Since fx is in pixels, and sensor_width_mm is known,
    # pixel_size_mm = sensor_width_mm / image_width_pixels
    # But we don't have image dimensions here. For simplicity, assume pixel_size is such that
    # we can compute directly, but actually, for FOV calculation, we need physical focal length.

    # Better way: FOV_x = 2 * atan(sensor_width_mm / (2 * (fx * pixel_size_mm)))
    # But pixel_size_mm = sensor_width_mm / image_width
    # So FOV_x = 2 * atan(image_width / (2 * fx))

    # Since we don't have image dimensions, let's assume standard values or calculate from K.
    # Actually, for undistorted FOV, we can use the formula with sensor dimensions.

    # Assuming the camera matrix is calibrated with sensor dimensions in mind.
    # For simplicity, let's compute FOV assuming the principal point and focal lengths.

    # Standard formula:
    # FOV_x = 2 * atan(sensor_width_mm / (2 * focal_length_mm))
    # But focal_length_mm = fx * pixel_size_mm
    # pixel_size_mm = sensor_width_mm / image_width_pixels

    # Since we don't have image_width, let's assume a typical value for Canon EOS R, image width ~ 6720 pixels for full res.
    # But to make it general, perhaps the script needs image dimensions.

    # For now, let's hardcode typical values. For Canon EOS, pixel size ~ 3.2 um = 0.0032 mm
    PIXEL_SIZE_MM = 0.0044  # approximate for Canon EOS

    focal_length_x_mm = fx * PIXEL_SIZE_MM
    focal_length_y_mm = fy * PIXEL_SIZE_MM

    fov_x_rad = 2 * np.arctan(sensor_width_mm / (2 * focal_length_x_mm))
    fov_y_rad = 2 * np.arctan(sensor_height_mm / (2 * focal_length_y_mm))

    fov_x_deg = np.degrees(fov_x_rad)
    fov_y_deg = np.degrees(fov_y_rad)

    return fov_x_deg, fov_y_deg

def get_fov(target_depth_cm):
    
    
    calibrations = load_calibrations("depth_groups")
    if not calibrations:
        print("No calibration files found")
        return

    try:
        calib = interpolate_calibration(calibrations, target_depth_cm)
        fov_x, fov_y = calculate_fov(calib['K'], SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM)

        print(f"At depth {target_depth_cm} cm:")
        print(f"FOV X: {fov_x:.2f} degrees")
        print(f"FOV Y: {fov_y:.2f} degrees")
        return fov_x, fov_y

    except ValueError as e:
        print(f"Error: {e}")