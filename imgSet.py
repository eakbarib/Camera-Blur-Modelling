import cv2 as cv
import cv2.aruco as ac
import exiv2
import numpy as np
import numpy.linalg as la
import json
import os
import matplotlib.pyplot as plt
from depthGroup import depth_groups

class imgSet:
    def __init__(self, folder, set_id, start, in_focus, end, extension='.JPG'):
        self.id = set_id
        self.start = start
        self.in_focus = in_focus - start
        self.count = end - start
        self.folder = folder
        self.extension = extension

    def read_meta(self, idx):
        """
        Returns an image's exif metadata
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for image set")
        
        filename = f"IMG_{self.start + idx:04d}{self.extension}"
        path = os.path.join(self.folder, self.id, filename)
        
        meta_img = exiv2.ImageFactory.open(path)
        meta_img.readMetadata()
        meta = meta_img.exifData()
    
        return meta
    
    def get_focus_distance_range(self, idx):
        """
        Returns a tuple of the min and max focus distances for an image
        """
        fmin = np.empty(self.count)
        fmax = np.empty(self.count)
        meta = self.read_meta(idx)
        fmin = meta["Exif.CanonFi.FocusDistanceLower"].getValue().toFloat()
        fmax = meta["Exif.CanonFi.FocusDistanceUpper"].getValue().toFloat()
        return (fmin, fmax)
        
    def read_img(self, idx):
        """
        Returns an image in the image set
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for image set")

        filename = f"IMG_{self.start + idx:04d}{self.extension}"
        path = os.path.join(self.folder, self.id, filename)

        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    def get_gt_path(self):
        """
        Returns the path to the calibration image used for this image set
        """
        return f"./calib_images/calib_{self.id}.png"
    
    def read_gt(self):
        """
        Returns the calibration image used for this image set
        """
        return cv.imread(self.get_gt_path())
    
    def get_pose(self):
        """
        Returns the pose of the plane for this image set
        """
        with open(f"{self.folder}/{self.id}/pose.json", 'r') as f:
            pose = json.load(f)
        return pose
        
    def get_calib(self, idx):
        """
        Returns the camera calibration for the depth of an image
        """
        low, high = self.get_focus_distance_range(idx)
        target_depth = round((low + high)*0.5, 0)
        
        # Find closest depth
        closest_depth = min(depth_groups, key=lambda depth_group: abs(depth_group.depth - target_depth))
        
        if closest_depth.calibration == {}:
            print(f"No calibration found for nearest depth to {target_depth}, ({closest_depth.depth})")
            return None
        return closest_depth.calibration
    
    def get_stack(self):
        """
        Returns a (count, height, width) matrix
        """
        return np.load(f"./stacks/{self.id}.npy")

        
bokeh_img_sets = {
    "or1_ir0_ds20": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or1_ir0_ds20", 9702, 9723, 9735),
    "or6_ir0_ds20": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or6_ir0_ds20", 9736, 9757, 9769),
    "or6_ir3_ds20": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or6_ir3_ds20", 9668, 9689, 9701),
    "or10_ir0_ds30": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or10_ir0_ds30", 9906, 9927, 9939),
    "or10_ir1_ds20": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or10_ir1_ds20", 9804, 9825, 9837),
    "or10_ir2_ds20": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or10_ir2_ds20", 9838, 9859, 9871),
    "or10_ir5_ds30": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or10_ir5_ds30", 9872, 9893, 9905),
    "or15_ir0_ds40": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or15_ir0_ds40", 9770, 9791, 9803),
    "or15_ir7_ds40": imgSet("./checkerboard & colmap/bokeh_calib_photos", "or15_ir7_ds40", 9940, 9961, 9973),
}

  # each set has 34 images, with the in-focus image at index 10 + first index, and the last image at index 33 + first index
    # and we have 113 sets
    # go into each directory in checkerboard & colmap/Focus Stack Calibration, and for each set, create an imgSet with the appropriate start, in_focus, and end indices
    # the appropriate start index is the index of the first image in the set, which you read from the filename in the directory, for example for directory 1 it is "IMG_5816.JPG", for directory 2 it is "IMG_5850.JPG", and so on and there might be some missing images, so you should read the directory and find the first image and the last image, and use those indices to create the imgSet
    # the in-focus image is always at index 10 + start index, and the last image is always at index 33 + start index, so you can use those indices to create the imgSet

checkerboard_img_sets = {
}

for dir_name in os.listdir("./checkerboard & colmap/Focus Stack Calibration"):
    dir_path = os.path.join("./checkerboard & colmap/Focus Stack Calibration", dir_name)
    if not os.path.isdir(dir_path):
        continue
    
    img_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
    if not img_files:
        continue
    
    img_files.sort()
    start_idx = int(img_files[0][4:8])  # Extract the number from "IMG_####.JPG"
    end_idx = int(img_files[-1][4:8])  # Extract the number from the last image file
    in_focus_idx = start_idx + 10  # In-focus image is always at index 10 + start index
    
    set_id = dir_name
    checkerboard_img_sets[set_id] = imgSet("./checkerboard & colmap/Focus Stack Calibration", set_id, start_idx, in_focus_idx, end_idx, '.JPG')
