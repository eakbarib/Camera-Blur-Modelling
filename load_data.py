import cv2 as cv
import cv2.aruco as ac
import matplotlib.pyplot as plt
import rawpy
import exiv2
import numpy as np
import json
from scipy.ndimage import convolve

class imgSet:
    def __init__(self, set_id, start, in_focus, end):
        self.id = set_id
        self.start = start
        self.in_focus = in_focus - start
        self.count = end - start
        
    def read_img(self, idx):
        """
        Returns an image in the image set and its exif metadata
        """
        if (0 > idx or idx >= self.count):
            raise ValueError("Index out of range for image set")
        
        path = f"./bokeh_calib_photos/{self.id}/IMG_{self.start + idx}.CR3"
        
        meta_img = exiv2.ImageFactory.open(path)
        meta_img.readMetadata()
        meta = meta_img.exifData()
        
        img = rawpy.imread(path)
        return (img.postprocess(), meta)
    
    def read_gt(self):
        """
        Returns the calibration image used for this image set
        """
        return cv.imread(f"./calib_images/calib_{self.id}.png")
    
    def get_stack(self):
        """
        Returns a (count, height, width) matrix
        """
        return np.load(f"./stacks/{self.id}.npy")
    
    def calc_pose(self):
        """
        Returns the orientation of the imaged plane with respect to the camera as a 3x4 affine matrix
        """
        # locate aruco patches
        
        img, meta = self.read_img(self.in_focus)
        gt_img = self.read_gt()
        
        detector = ac.ArucoDetector(ac.getPredefinedDictionary(ac.DICT_5X5_50))
        rects, ids, _ = detector.detectMarkers(img)
        plt.imshow(img)
        
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
        hmat, _ = cv.findHomography(src_points, dst_points)
        warped = cv.warpPerspective(gt_img, hmat, (img.shape[1], img.shape[0]))
        warped_alpha = cv.warpPerspective(np.full((gt_img.shape[0], gt_img.shape[1]), alpha, dtype=np.float32), hmat, (img.shape[1], img.shape[0]))
        plt.imshow((warped*warped_alpha[:,:,None] + img*(1 - warped_alpha)[:,:,None])/255)
        plt.show()
        
        # figure out how to convert from 3x3 homography matrix + focus distance to 3x4 affine matrix
        pass

img_sets = {
    "or6_ir0_ds20": imgSet("or6_ir0_ds20", 9575, 9625, 9646),
    "or6_ir3_ds20": imgSet("or6_ir3_ds20", 9503, 9553, 9574),
    "or10_ir0_ds30": imgSet("or10_ir0_ds30", 9431, 9481, 9502),
    "or10_ir1_ds20": imgSet("or10_ir1_ds20", 9359, 9409, 9430),
    "or10_ir2_ds20": imgSet("or10_ir2_ds20", 9287, 9337, 9358),
    "or10_ir5_ds30": imgSet("or10_ir5_ds30", 9215, 9265, 9286),
    "or15_ir0_ds40": imgSet("or15_ir0_ds40", 9143, 9193, 9214),
    "or15_ir7_ds40": imgSet("or15_ir7_ds40", 9071, 9121, 9142)
}

def gen_stack(set_id):
    img_set = img_sets[set_id]
    first, _ = img_set.read_img(0)
    stack = np.empty((img_set.count, first.shape[0], first.shape[1]), dtype=first.dtype)
    for i in range(img_set.count):
        img, _ = img_set.read_img(i)
        stack[i] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    np.save(f"./stacks/{img_set.id}.npy", stack)
    print(f"done stack for {img_set.id}")

# run this at some point before using imgSet.get_stack
def gen_stacks():
    #gen_stack("or6_ir0_ds20"),
    gen_stack("or6_ir3_ds20"),
    gen_stack("or10_ir0_ds30"),
    gen_stack("or10_ir1_ds20"),
    gen_stack("or10_ir2_ds20"),
    gen_stack("or10_ir5_ds30"),
    gen_stack("or15_ir0_ds40"),
    gen_stack("or15_ir7_ds40")

        
gen_stacks()

#img_sets["or6_ir0_ds20"].calc_pose()

sobel3d = [
    [[ 1, 2, 1],
     [ 2, 4, 2],
     [ 1, 2, 1]],
    [[ 0, 0, 0],
     [ 0, 0, 0],
     [ 0, 0, 0]],
    [[-1,-2,-1],
     [-2,-4,-2],
     [-1,-2,-1]]
]

stack = img_sets["or6_ir0_ds20"].get_stack()
dIdfocus = convolve(stack, sobel3d, mode='constant', cval=0.0, axes=(0,1,2))
plt.imshow(dIdfocus[20])
plt.show()