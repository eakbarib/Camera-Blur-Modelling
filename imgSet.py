import cv2 as cv
import json
from common import *

class imgSet:
    def __init__(self, folder, set_id, start, in_focus, end, extension='.JPG'):
        self.id = set_id
        self.start = start
        self.in_focus = in_focus - start
        self.count = end - start
        self.folder = folder
        self.extension = extension
        
    def get_img_path(self, idx):
        """Returns an image's path (as a Path object)"""
        return self.folder / f"IMG_{self.start + idx:04d}{self.extension}"
    
    def get_pattern_path(self):
        """
        Returns the path to the calibration image used for this image set
        """
        return f"./dot_patterns/calib_{self.id}.png"
    
    def get_focus_distance_range(self, idx):
        """Returns the min, max, and avg focus distance from image metadata"""
        return read_image_depth_range(self.get_img_path(idx))
        
    def read_img(self, idx):
        """
        Returns an image in the image set
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for image set")

        path = self.get_img_path(idx)

        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    def read_pattern(self):
        """
        Returns the dot pattern used for this image set
        """
        return cv.imread(self.get_pattern_path(), cv.IMREAD_GRAYSCALE)
    
    def get_pose(self):
        """
        Returns the pose of the plane for this image set
        """
        with open(self.folder / "pose.json", 'r') as f:
            pose = json.load(f)
        return pose
    
    def get_optim_calib(self, idx):
        """
        Returns the optimized calibration for this image
        """
        depth = self.get_focus_distance_range(idx)[2]
        optim_path = optimized_calibration_path / f"calib_{self.id}_{depth}.json"
        
        return load_calibration(optim_path)
        
    def get_calib(self, idx):
        """
        Returns the camera calibration for the depth of an image
        **May not be used in calibration pipeline
        """
        # lazy import if not already (it takes quite a while to generate depth groups)
        from depthGroup import depth_groups
        
        low, high = self.get_focus_distance_range(idx)
        target_depth = round((low + high)*0.5, 0)
        
        # Find closest depth
        closest_depth = min(depth_groups, key=lambda depth_group: abs(depth_group.depth - target_depth))
        
        if closest_depth.calibration == {}:
            print(f"No calibration found for nearest depth to {target_depth}, ({closest_depth.depth})")
            return None
        return closest_depth.calibration
        
dot_stack_sets = {
    "or1_ir0_ds20": imgSet(dot_stack_path / "or1_ir0_ds20", "or1_ir0_ds20", 9702, 9723, 9735),
    "or6_ir0_ds20": imgSet(dot_stack_path / "or6_ir0_ds20", "or6_ir0_ds20", 9736, 9757, 9769),
    "or6_ir3_ds20": imgSet(dot_stack_path / "or6_ir3_ds20", "or6_ir3_ds20", 9668, 9689, 9701),
    "or10_ir0_ds30": imgSet(dot_stack_path / "or10_ir0_ds30", "or10_ir0_ds30", 9906, 9927, 9939),
    "or10_ir1_ds20": imgSet(dot_stack_path / "or10_ir1_ds20", "or10_ir1_ds20", 9804, 9825, 9837),
    "or10_ir2_ds20": imgSet(dot_stack_path / "or10_ir2_ds20", "or10_ir2_ds20", 9838, 9859, 9871),
    "or10_ir5_ds30": imgSet(dot_stack_path / "or10_ir5_ds30", "or10_ir5_ds30", 9872, 9893, 9905),
    "or15_ir0_ds40": imgSet(dot_stack_path / "or15_ir0_ds40", "or15_ir0_ds40", 9770, 9791, 9803),
    "or15_ir7_ds40": imgSet(dot_stack_path / "or15_ir7_ds40", "or15_ir7_ds40", 9940, 9961, 9973),
}

  # each set has 34 images, with the in-focus image at index 10 + first index, and the last image at index 33 + first index
    # and we have 113 sets
    # go into each directory in checkerboard & colmap/Focus Stack Calibration, and for each set, create an imgSet with the appropriate start, in_focus, and end indices
    # the appropriate start index is the index of the first image in the set, which you read from the filename in the directory, for example for directory 1 it is "IMG_5816.JPG", for directory 2 it is "IMG_5850.JPG", and so on and there might be some missing images, so you should read the directory and find the first image and the last image, and use those indices to create the imgSet
    # the in-focus image is always at index 10 + start index, and the last image is always at index 33 + start index, so you can use those indices to create the imgSet