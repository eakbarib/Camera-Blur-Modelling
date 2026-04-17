import os
import json
import cv2 as cv

class depthGroup:
    def __init__(self, folder):
        self.folder = folder
        try:
            self.depth = float(folder.split("_")[-1])
        except ValueError:
            self = None
            return
        
        calib_path = os.path.join(folder, "calib.json")
        self.calibration = None
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                self.calibration = json.load(f)
             
        self.count = sum(1 for filename in os.listdir(folder) if filename.endswith(".JPG"))
        
    def read_img(self, idx):
        """_summary_
        Returns an image from this depth group
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for depth group")
        
        path = os.path.join(self.folder, f"{idx + 1}.JPG")
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
        

depth_groups = []
if os.path.exists("depth_groups"):
    for folder in os.listdir("depth_groups"):
        if folder.startswith("depth_"):
            group = depthGroup(os.path.join("depth_groups", folder))
            if group:
                depth_groups.append(group)
if not depth_groups:
    print(f"No depth folders found in depth_groups, please generate depth groups first.")