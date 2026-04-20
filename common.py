from pathlib import Path
import exiv2
import numpy as np
import json

paper_size_m = np.array((0.210, 0.297))
paper_size_px = np.array((2048, int(np.sqrt(2)*2048)))
m_per_px = paper_size_m[0]/paper_size_px[0]

dot_patterns_path = Path("dot_patterns")
dot_stack_path = Path("dot_stacks")
checkerboard_stack_path = Path("checkerboard_stacks")
checkerboard_single_path = Path("checkerboard_singles")
colmap_room_path = Path("colmap/room")
colmap_tabletop_path = Path("colmap/tabletop")
optimized_calibration_path = Path("optimized_calibrations")

# todo: 
# replace copying into depth groups with creating a registry in depthGroup.py
# fix naming conventions

def read_image_depth_range(image_path):
    """Returns the min, max, and avg focus distance from image metadata"""
    try:
        img = exiv2.ImageFactory.open(str(image_path))
        img.readMetadata()
        meta = img.exifData()
        low = meta["Exif.CanonFi.FocusDistanceLower"].getValue().toFloat()
        high = meta["Exif.CanonFi.FocusDistanceUpper"].getValue().toFloat()
        return low, high, (low + high)/2
    except Exception as e:
        print(f"Warning: could not read focus distance metadata for {image_path}: {e}")
        return None
    
def load_calibration(calib_path):
    if not calib_path.exists():
        return None
    with calib_path.open(mode='r') as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"])
    D = np.array(calib["distortion_coefficients"][0])
    return (K, D)
