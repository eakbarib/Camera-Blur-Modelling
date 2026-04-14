import cv2 as cv
import cv2.aruco as ac
import rawpy
import exiv2
import numpy as np
import numpy.linalg as la
import json
import os
import matplotlib.pyplot as plt
from depthGroup import depth_groups

class imgSet:
    def __init__(self, folder, set_id, start, in_focus, end):
        self.id = set_id
        self.start = start
        self.in_focus = in_focus - start
        self.count = end - start
        self.folder = folder

    def read_meta(self, idx):
        """
        Returns an image's exif metadata
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for image set")
        
        path = f"{self.folder}/{self.id}/IMG_{self.start + idx:04d}.CR3"
        
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
        
        path = f"{self.folder}/{self.id}/IMG_{self.start + idx:04d}.CR3"
        
        img = rawpy.imread(path)
        return img.postprocess()
    
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
    "or1_ir0_ds20": imgSet("./bokeh_calib_photos", "or1_ir0_ds20", 996, 1018, 1035),
    "or6_ir0_ds20": imgSet("./bokeh_calib_photos", "or6_ir0_ds20", 1036, 1058, 1074),
    "or6_ir3_ds20": imgSet("./bokeh_calib_photos", "or6_ir3_ds20", 1075, 1099, 1114),
    "or10_ir0_ds30": imgSet("./bokeh_calib_photos", "or10_ir0_ds30", 1115, 1137, 1154),
    "or10_ir1_ds20": imgSet("./bokeh_calib_photos", "or10_ir1_ds20", 1155, 1177, 1194),
    "or10_ir2_ds20": imgSet("./bokeh_calib_photos", "or10_ir2_ds20", 1195, 1217, 1234),
    "or10_ir5_ds30": imgSet("./bokeh_calib_photos", "or10_ir5_ds30", 1235, 1257, 1274),
    "or15_ir0_ds40": imgSet("./bokeh_calib_photos", "or15_ir0_ds40", 1275, 1297, 1314),
    "or15_ir7_ds40": imgSet("./bokeh_calib_photos", "or15_ir7_ds40", 1315, 1337, 1354),
}

checkerboard_img_sets = {
    1:imgSet("./checkerboard_images", "1", 1, 7, 19),
    2:imgSet("./checkerboard_images", "2", 20, 47, 59),
    3:imgSet("./checkerboard_images", "3", 60, 92, 100),
    4:imgSet("./checkerboard_images", "4", 101, 131, 141),
    5:imgSet("./checkerboard_images", "5", 142, 160, 182),
    6:imgSet("./checkerboard_images", "6", 183, 207, 223),
    7:imgSet("./checkerboard_images", "7", 306, 328, 346),
    8:imgSet("./checkerboard_images", "8", 388, 410, 428),
    9:imgSet("./checkerboard_images", "9", 429, 448, 469),
    10:imgSet("./checkerboard_images", "10", 470, 496, 510),
    11:imgSet("./checkerboard_images", "11", 511, 533, 551),
    12:imgSet("./checkerboard_images", "12", 552, 576, 592),
    13:imgSet("./checkerboard_images", "13", 593, 617, 633),
    14:imgSet("./checkerboard_images", "14", 634, 653, 674),
    15:imgSet("./checkerboard_images", "15", 675, 694, 715),
    16:imgSet("./checkerboard_images", "16", 716, 740, 755),
    17:imgSet("./checkerboard_images", "17", 756, 780, 795),
    18:imgSet("./checkerboard_images", "18", 796, 819, 835),
    19:imgSet("./checkerboard_images", "19", 836, 860, 875),
    20:imgSet("./checkerboard_images", "20", 916, 937, 955),
    21:imgSet("./checkerboard_images", "21", 956, 977, 995),
    22:imgSet("./checkerboard_images", "22", 8533, 8554, 8572),
    23:imgSet("./checkerboard_images", "23", 8586, 8594, 8613),
    24:imgSet("./checkerboard_images", "24", 8654, 8670, 8693),
    25:imgSet("./checkerboard_images", "25", 8694, 8713, 8733),
    26:imgSet("./checkerboard_images", "26", 8734, 8753, 8773),
    27:imgSet("./checkerboard_images", "27", 8774, 8793, 8813),
    28:imgSet("./checkerboard_images", "28", 8814, 8835, 8853),
    29:imgSet("./checkerboard_images", "29", 8854, 8873, 8893),
    30:imgSet("./checkerboard_images", "30", 8894, 8913, 8933),
    31:imgSet("./checkerboard_images", "31", 8992, 9013, 9033),
    32:imgSet("./checkerboard_images", "32", 9034, 9053, 9074),
    33:imgSet("./checkerboard_images", "33", 9075, 9094, 9115),
    34:imgSet("./checkerboard_images", "34", 9116, 9135, 9156),
    35:imgSet("./checkerboard_images", "35", 9157, 9176, 9197),
    36:imgSet("./checkerboard_images", "36", 9198, 9217, 9238),
    37:imgSet("./checkerboard_images", "37", 9239, 9258, 9279),
    38:imgSet("./checkerboard_images", "38", 9280, 9299, 9320),
    39:imgSet("./checkerboard_images", "39", 9321, 9340, 9361),
    40:imgSet("./checkerboard_images", "40", 9362, 9381, 9402),
    41:imgSet("./checkerboard_images", "41", 9403, 9422, 9443),
    42:imgSet("./checkerboard_images", "42", 9444, 9463, 9484),
    43:imgSet("./checkerboard_images", "43", 9526, 9544, 9566),
    44:imgSet("./checkerboard_images", "44", 9567, 9586, 9607),
    45:imgSet("./checkerboard_images", "45", 9608, 9627, 9648),
    46:imgSet("./checkerboard_images", "46", 9690, 9668, 9730),
    47:imgSet("./checkerboard_images", "47", 9731, 9750, 9771),
    48:imgSet("./checkerboard_images", "48", 9772, 9791, 9812),
    49:imgSet("./checkerboard_images", "49", 9813, 9832, 9853),
    50:imgSet("./checkerboard_images", "50", 9895, 9914, 9935),
    51:imgSet("./checkerboard_images", "51", 9936, 9955, 9999),
    52:imgSet("./checkerboard_images", "52", 224, 244, 264),
    53:imgSet("./checkerboard_images", "53", 265, 285, 305),
    54:imgSet("./checkerboard_images", "54", 347, 367, 387),
    55:imgSet("./checkerboard_images", "55", 876, 895, 915),
    56:imgSet("./checkerboard_images", "56", 8614, 8634, 8653),
    57:imgSet("./checkerboard_images", "57", 8946, 8965, 8991),
    58:imgSet("./checkerboard_images", "58", 9485, 9504, 9525),
    59:imgSet("./checkerboard_images", "59", 9649, 9668, 9689),
    60:imgSet("./checkerboard_images", "60", 9854, 9874, 9894),
}