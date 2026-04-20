import numpy as np
import cv2 as cv
import json
import cv2.aruco as ac
from common import *

def calc_pose(img_set):
    sheet_size = np.array(paper_size_m) # width, height in meters
    
    K, D = load_calibration(checkerboard_single_path / "calib.json")
    
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
    
    with open(dot_patterns_path / f"calib_{img_set.id}.json", 'r') as f:
        pattern_markers = json.load(f)
    
    num_pts = 4*len(rects)
    dst_points = np.empty((num_pts,2), dtype=np.float32)
    src_points = np.empty((num_pts,3), dtype=np.float32)
    
    # iterate over detected markers
    i = 0
    for j in range(len(ids)):
        marker = next(marker for marker in pattern_markers if marker["id"] == ids[j])
        origin = np.array(marker["origin"])
        size = np.array(marker["size"])
        
        corners = origin + np.array([[0,0], [size,0], [size,size], [0,size]])
        world_corners = corners*m_per_px - sheet_size*0.5
        
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