import cv2 as cv
import cv2.aruco as ac
import numpy as np
import matplotlib.pyplot as plt
import json
from common import *

def gen_calib_image(true_marker_size, true_dot_outer_radius, true_dot_inner_radius, true_dot_spacing):
    """
    Generates a calibration image and saves it to `dot_patterns/`
    """
    width_mm = paper_size_m[0]*1000
    
    marker_dict = ac.getPredefinedDictionary(ac.DICT_5X5_50)

    dest_width = paper_size_px[0]
    dest_height = paper_size_px[1]

    conversion_factor = dest_width/width_mm

    calib_image = np.full((dest_height, dest_width), 255, dtype=np.uint8)

    # generate dot grid
    dot_outer_radius = int(true_dot_outer_radius*conversion_factor)
    dot_inner_radius = int(true_dot_inner_radius*conversion_factor)
    dot_padding = int(true_dot_outer_radius*2*conversion_factor)
    dot_spacing = int(true_dot_spacing*conversion_factor)

    w_dots = (dest_width - 2*dot_padding)//dot_spacing + 1
    h_dots = (dest_height - 2*dot_padding)//dot_spacing + 1
    for x in range(w_dots):
        for y in range(h_dots):
            dot = np.full((dot_outer_radius*2, dot_outer_radius*2), 255, dtype=np.uint8)
            coords = np.indices((dot_outer_radius*2, dot_outer_radius*2)) - dot_outer_radius
            dot[np.linalg.norm(coords,axis=0) < dot_outer_radius] = 0
            dot[np.linalg.norm(coords,axis=0) < dot_inner_radius] = 255
            
            dot_pos = np.array((y, x))*dot_spacing + dot_padding - dot_outer_radius
            s = np.maximum(dot_pos, (0,0))
            e = np.minimum(dot_pos + dot_outer_radius*2, (dest_height, dest_width))
            
            a = s - dot_pos
            b = a + e - s
            
            calib_image[s[0]:e[0], s[1]:e[1]] = dot[a[0]:b[0], a[1]:b[1]]

    # generate aruco grid
    marker_size = int(true_marker_size*conversion_factor)
    marker_padding = int(true_marker_size*conversion_factor)
    marker_spacing = int((dest_width - 2*marker_padding)/3)
    marker_expand = int(marker_size/7)
    marker_outer_size = int(marker_size + 2*marker_expand)
    
    markers = []

    w_markers = (dest_width - 2*marker_padding)//marker_spacing + 1
    h_markers = (dest_height - 2*marker_padding)//marker_spacing + 1
    for x in range(w_markers):
        for y in range(h_markers):
            marker_id = x + y*w_markers
            marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
            ac.generateImageMarker(marker_dict, marker_id, marker_size, marker)
            padded_marker = np.full((marker_outer_size, marker_outer_size), 255, dtype=np.uint8)
            padded_marker[marker_expand:marker_size + marker_expand, marker_expand:marker_size + marker_expand] = marker
            
            marker_pos = np.array((y, x))*marker_spacing + marker_padding - marker_outer_size//2
            s = np.maximum(marker_pos, (0,0))
            e = np.minimum(marker_pos + marker_outer_size, (dest_height, dest_width))
            
            a = s - marker_pos
            b = a + e - s
            
            calib_image[s[0]:e[0], s[1]:e[1]] = padded_marker[a[0]:b[0], a[1]:b[1]]
            markers.append({"id":int(marker_id), "origin":[int(marker_pos[1]), int(marker_pos[0])], "size":int(marker_outer_size)})

    # debug display
    #plt.imshow(calib_image)
    #for marker in markers:
    #    origin = np.array(marker["origin"])
    #    plt.plot(origin[0], origin[1], 'x')
    #plt.show()

    filename = f"calib_or{true_dot_outer_radius}_ir{true_dot_inner_radius}_ds{true_dot_spacing}"
    # write image
    cv.imwrite(dot_patterns_path / f"{filename}.png", calib_image)
    # write marker positions
    with open(dot_patterns_path / f"{filename}.json", 'w') as f:
        json.dump(markers, f, indent=4)

gen_calib_image(15, 15, 7, 40)
gen_calib_image(15, 15, 0, 40)
gen_calib_image(15, 10, 5, 30)
gen_calib_image(15, 10, 0, 30)
gen_calib_image(15, 6, 3, 20)
gen_calib_image(15, 6, 0, 20)
gen_calib_image(15, 10, 2, 20)
gen_calib_image(15, 10, 1, 20)