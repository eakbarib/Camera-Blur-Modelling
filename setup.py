from imgSet import bokeh_img_sets
import numpy as np
import cv2 as cv
import json
import cv2.aruco as ac

# pre-generates a focus stack (significantly speeds up focus stack handling)

def gen_stack(set_id):
    img_set = bokeh_img_sets[set_id]
    first, _ = img_set.read_img(0)
    stack = np.empty((img_set.count, first.shape[0], first.shape[1]), dtype=first.dtype)
    for i in range(img_set.count):
        img, _ = img_set.read_img(i)
        stack[i] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    np.save(f"./stacks/{img_set.id}.npy", stack)
    print(f"done stack for {img_set.id}")

# calculates the plane pose of an image set

def calc_pose(img_set):
    gt_img = img_set.read_gt()
    
    sheet_size = np.array([0.210, 0.297]) # width, height in meters
    meters_per_pixel = sheet_size[0]/gt_img.shape[1] # paper width (meters) / calib image width (pixels)
    
    calib = img_set.get_calib(img_set.in_focus)
    K = np.array(calib["camera_matrix"])
    D = np.array(calib["distortion_coefficients"])
    
    # locate aruco markers
    
    img = img_set.read_img(img_set.in_focus)
    
    # detect
    detector = ac.ArucoDetector(ac.getPredefinedDictionary(ac.DICT_5X5_50))
    rects, ids, _ = detector.detectMarkers(img)
    
    if ids is None:
        print("No markers detected")
        return None
    
    # refine
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    for rect in rects:
        cv.cornerSubPix(gray, rect, (5, 5), (-1, -1), criteria)
    
    # get corresponding points
    
    with open(f"./calib_images/calib_{img_set.id}.json", 'r') as f:
        gt_markers = json.load(f)
    
    num_pts = 4*len(rects)
    dst_points = np.empty((num_pts,2), dtype=np.float32)
    src_points = np.empty((num_pts,3), dtype=np.float32)
    
    # iterate over detected markers
    i = 0
    for j in range(len(ids)):
        marker = next(marker for marker in gt_markers if marker["id"] == ids[j])
        origin = np.array(marker["origin"])
        size = np.array(marker["size"])
        
        corners = origin + np.array([[0,0], [size,0], [size,size], [0,size]])
        world_corners = corners*meters_per_pixel - sheet_size*0.5
        
        src_points[i:i+4] = np.concat((world_corners, np.zeros((4,1))), axis=1)
        dst_points[i:i+4] = rects[j][0]
        i += 4
    
    # find pose
    
    ret, rvec, t = cv.solvePnP(src_points, dst_points, K, D)
    if not ret:
        print("Failed to find transform from src to dst")
        return None
    
    # convert rotation vector to matrix
    R, _ = cv.Rodrigues(rvec)
    # build plane pose matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:,0]
    
    # save result
    
    result = {
        "plane_pose": T.tolist()
    }
    with open(f"{img_set.folder}/{img_set.id}/pose.json", 'w') as f:
        json.dump(result, f, indent=4)
    

def gen_stacks():
    gen_stack("or6_ir0_ds20"),
    gen_stack("or6_ir3_ds20"),
    gen_stack("or10_ir0_ds30"),
    gen_stack("or10_ir1_ds20"),
    gen_stack("or10_ir2_ds20"),
    gen_stack("or10_ir5_ds30"),
    gen_stack("or15_ir0_ds40"),
    gen_stack("or15_ir7_ds40")
    
#gen_stacks()

# calculate and save poses for image sets
def calc_poses():
    for img_set in bokeh_img_sets.values():
        print(f"Taking pose of {img_set.id}")
        calc_pose(img_set)
        
calc_poses()