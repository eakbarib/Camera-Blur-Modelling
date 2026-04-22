import json
import cv2 as cv
from common import *

class depthGroup:
    def __init__(self, depth):
        self.depth = depth
        self.image_paths = []
        self.count = 0
        
    def add_image(self, path):
        """Adds an image to the group"""
        self.image_paths.append(path)
        self.count += 1
    
    def read_calibration(self):
        """Loads the calibration for this group"""
        calib_file_path = checkerboard_stack_path / "calibs.json"
        if calib_file_path.exists():
            with calib_file_path.open(mode='r') as file:
                calibs = json.load(file)
            return calibs.get(str(self.depth), None)
        else:
            raise Exception("Calibrations for depth groups missing. Run init.py first.")
        
    def read_img(self, idx):
        """Returns an image from this depth group"""
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for depth group")
        
        img = cv.imread(self.image_paths[idx], cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {self.image_paths[idx]}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

# depth bin tolerance
DEPTH_ROUND_DECIMALS = 6

print("generating depth groups...")
depth_groups = []
# iterate over stacks
for folder_path in checkerboard_stack_path.iterdir():
    if folder_path.is_dir():
        # iterate over stack
        for image_path in folder_path.iterdir():
            if (image_path.suffix.lower() in ['.jpg', '.jpeg']):
                depth_bin = round(read_image_depth_range(image_path)[2])
                # try select depth group with same depth
                group = next((group for group in depth_groups if group.depth == depth_bin), None)
                if group == None:
                    group = depthGroup(depth_bin)
                    depth_groups.append(group)
                group.add_image(image_path)
print("done.")