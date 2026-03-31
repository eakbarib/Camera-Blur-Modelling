import cv2 as cv
import cv2.aruco as ac
import rawpy
import exiv2
import numpy as np
import numpy.linalg as la
import json
import os
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
        Returns the pose for this image set
        """
        with open('pose_estimations.json', 'r') as f:
            poses = json.load(f)
        return poses[self.id]
        
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
    
    def calc_homography(self):
        """
        Returns the orientation of the imaged plane with respect to the camera as a 3x3 homography matrix
        """
        # locate aruco patches
        
        img = self.read_img(self.in_focus)
        gt_img = self.read_gt()
        
        detector = ac.ArucoDetector(ac.getPredefinedDictionary(ac.DICT_5X5_50))
        rects, ids, _ = detector.detectMarkers(img)
        
        # find homography between real image and calibration image
        
        with open(f"./calib_images/calib_{self.id}.json", 'r') as f:
            gt_markers = json.load(f)
        
        dst_points = np.empty((4*len(gt_markers),2), dtype=rects[0].dtype)
        src_points = np.empty((4*len(gt_markers),2), dtype=rects[0].dtype)
        
        i = 0
        for marker in gt_markers:
            origin = np.array(marker["origin"])
            size = np.array(marker["size"])
            src_points[i:i+4] = origin + np.array([[0,0], [size,0], [size,size], [0,size]])
            dst_points[i:i+4] = rects[np.where(ids == marker["id"])[0][0]][0]
            i += 4
        
        # debugging display
        alpha = 0.5
        H, _ = cv.findHomography(src_points, dst_points)
        #warped = cv.warpPerspective(gt_img, H, (img.shape[1], img.shape[0]))
        #warped_alpha = cv.warpPerspective(np.full((gt_img.shape[0], gt_img.shape[1]), alpha, dtype=np.float32), H, (img.shape[1], img.shape[0]))
        #plt.imshow((warped*warped_alpha[:,:,None] + img*(1 - warped_alpha)[:,:,None])/255)
        #plt.show()
        
        return H

bokeh_img_sets = {
    "or1_ir0_ds20":imgSet("./bokeh_calib_photos", "or1_ir0_ds20", 996, 1018, 1035),
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
}

#img_sets["or6_ir0_ds20"].calc_pose()