import os
import json
import rawpy

class depthGroup:
    def __init__(self, folder):
        self.folder = folder
        self.depth = float(folder.split("_")[1])
        
        calib_path = os.path.join(folder, "calib.json")
        self.calibration = None
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                self.calibration = json.load(f)
             
        self.count = sum(1 for filename in os.listdir(folder) if filename.endswith(".CR3"))
    
    def __init__(self, folder, depth, count, calibration=None):
        self.folder = folder
        self.depth = depth
        self.count = count
        self.calibration = calibration
        
    def read_img(self, idx):
        """_summary_
        Returns an image from this depth group
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for depth group")
        
        path = os.path.join(self.folder, f"{idx + 1}.CR3")
        img = rawpy.imread(path)
        return img.postprocess()
        

depth_groups = []
if os.path.exists("depth_groups"):
    for folder in os.listdir("depth_groups"):
        if folder.startswith("depth_"):
            try:
                depth_groups.append(depthGroup(folder))
            except ValueError:
                continue
if not depth_groups:
    print(f"No depth folders found in depth_groups, please generate depth groups first.")