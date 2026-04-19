from pathlib import Path
import argparse
import json
import numpy as np
import sqlite3
from PIL import Image
import exiv2
import os
import subprocess
import sys

# Path to COLMAP binary
COLMAP_BIN_PATH = Path("colmap-bin/COLMAP.bat") if os.name == 'nt' else Path("colmap-bin/colmap")

# given a set of images and optimized camera matrices, reconstruct 3D points and calculate reprojection errors


def run_colmap_command(command_args, cwd=None):
    """
    Run a COLMAP command using the executable from colmap-bin folder.

    Args:
        command_args: List of command arguments
        cwd: Working directory for the command

    Returns:
        bool: True if command succeeded, False otherwise
    """
    if not COLMAP_BIN_PATH.exists():
        print(f"COLMAP binary not found at {COLMAP_BIN_PATH}")
        return False

    cmd = [str(COLMAP_BIN_PATH)] + command_args
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print("Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def get_jpg_image_paths(image_dir):
    """Get sorted list of JPG image paths from a directory."""
    image_dir = Path(image_dir)
    return sorted([str(f) for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])


def run_feature_extraction(database_path, image_dir, single_camera_per_image=False):
    """Run COLMAP feature extraction with GPU acceleration."""
    args = [
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0"
    ]
    if single_camera_per_image:
        args += ["--ImageReader.single_camera_per_image", "1"]
    return run_colmap_command(args)


def run_exhaustive_matching(database_path):
    """Run COLMAP exhaustive matching with GPU acceleration."""
    return run_colmap_command([
        "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0"
    ])


def run_mapper(database_path, image_dir, output_dir, fix_intrinsics=False):
    """Run COLMAP incremental mapper with GPU acceleration."""
    args = [
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(output_dir),
        "--Mapper.ba_use_gpu", "1",
        "--Mapper.ba_gpu_index", "0"
    ]
    if fix_intrinsics:
        args += [
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0"
        ]
    return run_colmap_command(args)


def convert_model_to_txt(output_dir, sparse_model_path="sparse/0"):
    """
    Convert COLMAP binary model to text format.

    Args:
        output_dir: Path to the output directory from reconstruction
        sparse_model_path: Relative path to sparse model within output_dir (default: sparse/0)

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    output_dir = Path(output_dir)
    input_path = output_dir / sparse_model_path
    output_path = output_dir / sparse_model_path  # Convert in place

    if not input_path.exists():
        print(f"Sparse model not found at {input_path}")
        return False

    success = run_colmap_command([
        "model_converter",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--output_type", "TXT"
    ])

    if success:
        print(f"Model converted to text format at {output_path}")
        return True
    else:
        print("Model conversion failed")
        return False


def quaternion_to_rotation_matrix(qvec):
    """Convert COLMAP quaternion [qw, qx, qy, qz] to a rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=float)


def project_colmap_point(point3d, qvec, tvec, camera):
    """Project a 3D point using COLMAP camera parameters."""
    R = quaternion_to_rotation_matrix(qvec)
    t = np.array(tvec, dtype=float).reshape(3)
    X = np.array(point3d, dtype=float).reshape(3)
    X_cam = R @ X + t

    if X_cam[2] <= 0:
        return np.array([np.nan, np.nan], dtype=float)

    x_norm = X_cam[0] / X_cam[2]
    y_norm = X_cam[1] / X_cam[2]

    model = camera["model"]
    params = camera["params"]

    if model == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        x_dist = x_norm
        y_dist = y_norm
        return np.array([fx * x_dist + cx, fx * y_dist + cy], dtype=float)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        x_dist = x_norm
        y_dist = y_norm
        return np.array([fx * x_dist + cx, fy * y_dist + cy], dtype=float)
    elif model == "SIMPLE_RADIAL":
        fx, cx, cy, k = params
        r2 = x_norm**2 + y_norm**2
        radial = 1 + k * r2
        x_dist = x_norm * radial
        y_dist = y_norm * radial
        return np.array([fx * x_dist + cx, fx * y_dist + cy], dtype=float)
    elif model == "RADIAL":
        fx, fy, cx, cy, k1, k2 = params
        r2 = x_norm**2 + y_norm**2
        radial = 1 + k1 * r2 + k2 * r2**2
        x_dist = x_norm * radial
        y_dist = y_norm * radial
        return np.array([fx * x_dist + cx, fy * y_dist + cy], dtype=float)
    elif model in {"OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[:9]
        r2 = x_norm**2 + y_norm**2
        if model == "OPENCV_FISHEYE":
            theta = np.sqrt(r2)
            if theta > 1e-8:
                theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6)
                scale = theta_d / theta
                x_dist = x_norm * scale
                y_dist = y_norm * scale
            else:
                x_dist = x_norm
                y_dist = y_norm
        else:
            radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
            y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
        return np.array([fx * x_dist + cx, fy * y_dist + cy], dtype=float)
    else:
        raise ValueError(f"Unsupported camera model: {model}")


def parse_colmap_cameras_txt(cameras_path):
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def parse_colmap_images_txt(images_path):
    images = {}
    with open(images_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for i in range(0, len(lines), 2):
        header = lines[i].split()
        image_id = int(header[0])
        qvec = list(map(float, header[1:5]))
        tvec = list(map(float, header[5:8]))
        camera_id = int(header[8])
        name = header[9]

        observations = []
        obs_parts = lines[i + 1].split()
        for j in range(0, len(obs_parts), 3):
            x = float(obs_parts[j])
            y = float(obs_parts[j + 1])
            point3d_id = int(obs_parts[j + 2])
            observations.append({
                "xy": np.array([x, y], dtype=float),
                "point3d_id": point3d_id
            })

        images[image_id] = {
            "name": name,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "observations": observations
        }
    return images


def parse_colmap_points3d_txt(points_path):
    points3D = {}
    with open(points_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = list(map(float, parts[1:4]))
            points3D[point_id] = xyz
    return points3D


def calculate_reprojection_error(output_dir, sparse_model_path="sparse/0"):
    """
    Calculate the reprojection error from a COLMAP TXT model.

    Args:
        output_dir: Directory containing the COLMAP reconstruction output.
        sparse_model_path: Relative path inside output_dir to the sparse model (default: sparse/0).

    Returns:
        dict: reprojection error statistics.
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / sparse_model_path

    cameras_path = model_dir / "cameras.txt"
    images_path = model_dir / "images.txt"
    points_path = model_dir / "points3D.txt"

    if not cameras_path.exists() or not images_path.exists() or not points_path.exists():
        print("COLMAP TXT model files not found. Please run model_converter first.")
        return None

    cameras = parse_colmap_cameras_txt(cameras_path)
    images = parse_colmap_images_txt(images_path)
    points3D = parse_colmap_points3d_txt(points_path)

    errors = []
    for image in images.values():
        camera = cameras[image["camera_id"]]
        for obs in image["observations"]:
            point_id = obs["point3d_id"]
            if point_id < 0 or point_id not in points3D:
                continue
            projected = project_colmap_point(points3D[point_id], image["qvec"], image["tvec"], camera)
            if np.isnan(projected).any():
                continue
            errors.append(np.linalg.norm(projected - obs["xy"]))

    if len(errors) == 0:
        print("No valid reprojection error measurements found.")
        return None

    errors = np.array(errors)
    stats = {
        "num_observations": int(errors.size),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "std_error": float(np.std(errors))
    }

    print(f"Reprojection error statistics: {stats}")
    return stats


def reconstruct_3d_simple(image_dir, output_dir, database_name="reconstruction.db"):
    """
    Simpler 3D reconstruction using COLMAP's default functions without manual database setup.
    Uses COLMAP's automatic camera model selection and intrinsic estimation with GPU acceleration.

    Args:
        image_dir: Path to directory containing images
        output_dir: Path to output directory for results
        database_name: Name of the database file to create

    Returns:
        bool: True if reconstruction succeeded, False otherwise
    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(exist_ok=True)

    database_path = output_dir / database_name

    if not run_feature_extraction(database_path, image_dir):
        print("Feature extraction failed")
        return False

    if not run_exhaustive_matching(database_path):
        print("Feature matching failed")
        return False

    if not run_mapper(database_path, image_dir, output_dir):
        print("Reconstruction failed")
        return False

    print("Reconstruction completed successfully")
    return True


def load_optimized_camera_matrices(opt_dir="optimized_camera_matrix"):
    """Load optimized camera matrices from JSON files, including depth metadata."""
    opt_dir = Path(opt_dir)
    cameras = []

    if not opt_dir.exists():
        print(f"Optimized camera directory not found: {opt_dir}")
        return cameras

    for path in sorted(opt_dir.glob("optimized_matrix_idx_*.json")):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        target_idx = data.get('target_idx')
        depth_m = data.get('depth_m')
        camera_matrix = np.array(data.get('camera_matrix', []), dtype=float)
        if target_idx is None or depth_m is None or camera_matrix.shape != (3, 3):
            continue

        cameras.append({
            "target_idx": int(target_idx),
            "depth_m": float(depth_m),
            "camera_matrix": camera_matrix
        })

    return cameras


def camera_params_from_matrix(camera_matrix, model="OPENCV"):
    """Create COLMAP camera parameters from a 3x3 intrinsic matrix."""
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if model == "OPENCV":
        # Use zero distortion if not available in the optimized JSON.
        return [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif model == "PINHOLE":
        return [fx, fy, cx, cy]
    else:
        raise ValueError(f"Unsupported camera model for params generation: {model}")


def get_focus_distance_range_from_metadata(image_path):
    """Read image metadata and return the focus distance bounds."""
    try:
        img = exiv2.ImageFactory.open(str(image_path))
        img.readMetadata()
        meta = img.exifData()
        low = meta["Exif.CanonFi.FocusDistanceLower"].getValue().toFloat()
        high = meta["Exif.CanonFi.FocusDistanceUpper"].getValue().toFloat()
        return float(low), float(high)
    except Exception as e:
        print(f"Warning: could not read focus distance metadata for {image_path}: {e}")
        return None


def get_average_depth_from_metadata(image_path):
    """Calculate the average depth from focus distance metadata."""
    depth_range = get_focus_distance_range_from_metadata(image_path)
    if depth_range is None:
        return None
    low, high = depth_range
    return (low + high) / 2.0


def find_closest_optimized_camera_by_depth(depth, optimized_cameras):
    """Find the optimized camera entry with the closest depth_m value."""
    if depth is None or not optimized_cameras:
        return None
    return min(optimized_cameras, key=lambda c: abs(c["depth_m"] - depth))


def insert_camera_to_db(database_path, camera_id, model, width, height, params):
    """Insert a camera into the COLMAP database cameras table."""
    model_map = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 8
    }
    model_id = model_map.get(model, 1)
    params_blob = np.array(params, dtype=np.float64).tobytes()
    conn = sqlite3.connect(str(database_path))
    try:
        conn.execute("""
            INSERT OR REPLACE INTO cameras (camera_id, model, width, height, params, prior_focal_length)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (camera_id, model_id, width, height, params_blob, 1))
        conn.commit()
    finally:
        conn.close()


def update_image_camera_in_db(database_path, image_name, camera_id):
    """Update the camera_id for an image in the COLMAP database images table."""
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.execute("UPDATE images SET camera_id = ? WHERE name = ?", (camera_id, image_name))
        conn.commit()
        if cur.rowcount == 0:
            print(f"Warning: image '{image_name}' not found in database images table.")
            return False
        return True
    finally:
        conn.close()


def ensure_colmap_db_initialized(database_path, image_dir):
    """Ensure the COLMAP database exists and has the required tables."""
    if not Path(database_path).exists():
        print(f"Database {database_path} does not exist. Initializing with COLMAP feature_extractor...")
        run_feature_extraction(database_path, image_dir, single_camera_per_image=True)
    else:
        # Check for cameras table
        conn = sqlite3.connect(str(database_path))
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")
            if cur.fetchone() is None:
                print(f"Database {database_path} missing cameras table. Initializing with COLMAP feature_extractor...")
                run_feature_extraction(database_path, image_dir, single_camera_per_image=True)
        finally:
            conn.close()


def reconstruct_3d(database_name, output_dir, image_dir):
    """
    3D reconstruction using optimized camera matrices from optimized_camera_matrix JSON files.

    Args:
        database_name: Name of the database file
        output_dir: Output directory path
        image_dir: Directory containing images

    Returns:
        bool: True if reconstruction succeeded, False otherwise
    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(exist_ok=True)

    database_path = output_dir / database_name
    optimized_cameras = load_optimized_camera_matrices("optimized_camera_matrix")

    image_paths = get_jpg_image_paths(image_dir)
    if not image_paths:
        print(f"No JPG images found in {image_dir}")
        return False

    if not optimized_cameras:
        print("No optimized camera matrices found. Cannot proceed.")
        return False

    cam_type = "PINHOLE"

    # Run feature extraction first to populate the database with images and features.
    if not run_feature_extraction(database_path, image_dir, single_camera_per_image=True):
        print("Feature extraction failed")
        return False

    # Replace auto-generated cameras with our optimized ones.
    optimized_cameras = sorted(optimized_cameras, key=lambda c: c["depth_m"])

    with Image.open(image_paths[0]) as base_img:
        width, height = base_img.size

    # Clear all auto-generated cameras and insert only optimized ones.
    conn = sqlite3.connect(str(database_path))
    try:
        conn.execute("DELETE FROM cameras")
        conn.commit()
    finally:
        conn.close()

    for cam_id, camera_entry in enumerate(optimized_cameras, start=1):
        params = camera_params_from_matrix(camera_entry["camera_matrix"], cam_type)
        insert_camera_to_db(database_path, cam_id, cam_type, width, height, params)
        camera_entry["camera_id"] = cam_id

    # Assign each image to the closest optimized camera by depth.
    for i, img_path in enumerate(image_paths):
        image_depth = get_average_depth_from_metadata(img_path)
        camera_entry = None

        if image_depth is not None:
            camera_entry = find_closest_optimized_camera_by_depth(image_depth, optimized_cameras)
            if camera_entry is not None:
                print(f"Matched {img_path} depth {image_depth:.1f} m to optimized depth {camera_entry['depth_m']:.1f} m")
        else:
            print(f"Warning: depth metadata missing for image {img_path}. Falling back to target index.")

        if camera_entry is None:
            camera_entry = next((c for c in optimized_cameras if c["target_idx"] == i), None)
            if camera_entry is not None:
                print(f"Fallback matched {img_path} to optimized camera index {i}")

        if camera_entry is None:
            print(f"Warning: No optimized matrix found for image {img_path}. Skipping this image.")
            continue

        camera_id = camera_entry["camera_id"]

        image_name = Path(img_path).name
        if not update_image_camera_in_db(database_path, image_name, camera_id):
            print(f"Failed to update camera for image {img_path}")
            return False

    # Rebuild rigs/frames/frame_data to match the new camera assignments.
    conn = sqlite3.connect(str(database_path))
    try:
        # Clear existing rig structure (cascades to rig_sensors, frames, frame_data)
        conn.execute("DELETE FROM frame_data")
        conn.execute("DELETE FROM frames")
        conn.execute("DELETE FROM rig_sensors")
        conn.execute("DELETE FROM rigs")
        # Reset autoincrement counters
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('rigs', 'frames')")

        # Create one rig per unique camera_id
        unique_cam_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT camera_id FROM images ORDER BY camera_id"
        ).fetchall()]
        rig_map = {}  # camera_id -> rig_id
        for cam_id in unique_cam_ids:
            conn.execute(
                "INSERT INTO rigs (ref_sensor_id, ref_sensor_type) VALUES (?, 0)",
                (cam_id,)
            )
            rig_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            rig_map[cam_id] = rig_id

        # Create one frame per image, linked to the rig for its camera
        rows = conn.execute("SELECT image_id, camera_id FROM images ORDER BY image_id").fetchall()
        for image_id, cam_id in rows:
            rig_id = rig_map[cam_id]
            conn.execute("INSERT INTO frames (rig_id) VALUES (?)", (rig_id,))
            frame_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, 0)",
                (frame_id, image_id, cam_id)
            )

        conn.commit()
        print(f"Rebuilt rig structure: {len(rig_map)} rigs for {len(rows)} images")
    finally:
        conn.close()

    if not run_exhaustive_matching(database_path):
        print("Feature matching failed")
        return False

    if not run_mapper(database_path, image_dir, output_dir, fix_intrinsics=True):
        print("Reconstruction failed")
        return False

    return True


def main(use_simple=False, output_dir=None, image_folder=None):
    """
    Main function to run 3D reconstruction.

    Args:
        use_simple: If True, uses simple reconstruction (default COLMAP settings).
                   If False, uses custom reconstruction with manual camera setup.
        output_dir: Output directory for COLMAP results (optional, overrides default)
        image_folder: Image folder for reconstruction (optional, overrides default)

    Returns:
        bool: True if reconstruction succeeded, False otherwise
    """
    output_dir = Path(output_dir) if output_dir is not None else Path("colmap/")
    image_folder = Path(image_folder) if image_folder is not None else Path('COLMAP_SCENE/')

    # convert_model_to_txt(output_dir)  # Convert model to text format for easier analysis
    # return True  # Exit after conversion for testing purposes

    # Check if COLMAP binary exists
    if not COLMAP_BIN_PATH.exists():
        print(f"Error: COLMAP binary not found at {COLMAP_BIN_PATH}")
        print("Please ensure COLMAP is installed in the colmap-bin folder")
        return False

    # Check if image folder exists and has images
    if not image_folder.exists():
        print(f"Error: Image folder {image_folder} does not exist")
        return False

    image_paths = get_jpg_image_paths(image_folder)
    if not image_paths:
        print(f"Error: No JPG images found in {image_folder}")
        return False

    print(f"Found {len(image_paths)} images in {image_folder}")

    if use_simple:
        print("Running simple reconstruction with GPU acceleration...")
        success = reconstruct_3d_simple(image_folder, output_dir)
    else:
        print("Running custom reconstruction with manual camera setup and GPU acceleration...")
        success = reconstruct_3d("base_recon.db", output_dir, image_folder)

    if success:
        print("3D reconstruction completed successfully!")
        return True
    else:
        print("3D reconstruction failed!")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="COLMAP reconstruction and reprojection tools")
    parser.add_argument("--use_simple", action="store_true", help="Run simple reconstruction using COLMAP default settings")
    parser.add_argument("--output_dir", default="colmap/", help="COLMAP output directory")
    parser.add_argument("--image_folder", default="COLMAP_SCENE/", help="Image folder for reconstruction")
    parser.add_argument("--reprojection", action="store_true", help="Calculate reprojection error for existing COLMAP TXT output")
    parser.add_argument("--sparse_model_path", default="sparse/0", help="Relative sparse model path inside output_dir")
    parser.add_argument("--convert_text", action="store_true", help="Convert existing COLMAP binary model to TXT before reprojection")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.reprojection:
        if args.convert_text:
            converted = convert_model_to_txt(args.output_dir, args.sparse_model_path)
            if not converted:
                sys.exit(1)
        stats = calculate_reprojection_error(args.output_dir, args.sparse_model_path)
        if stats is None:
            sys.exit(1)

        output_file = Path(args.output_dir) / "reprojection_error.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Reprojection error statistics:\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Reprojection error statistics written to {output_file}")
        sys.exit(0)

    success = main(use_simple=args.use_simple, output_dir=args.output_dir, image_folder=args.image_folder)
    sys.exit(0 if success else 1)