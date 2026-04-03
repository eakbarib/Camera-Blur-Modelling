import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
from imgSet import bokeh_img_sets
import matplotlib.pyplot as plt
import Camera

# todo: fix scaling
# try direct rendering mode

def create_mitsuba_scene(cam_pose, calib_image_path, cam_mat, distortion, focus_distance, render_size):
    """Create Mitsuba scene dictionary"""
    
    scene_dict = {
        'type': 'scene',
        
        'sensor': {
            'type': 'distorted_camera',
            'fx': cam_mat[0,0],
            'fy': cam_mat[1,1],
            'cx': cam_mat[0,2],
            'cy': cam_mat[1,2],
            'k1': distortion[0],
            'k2': distortion[1],
            'p1': distortion[2],
            'p2': distortion[3],
            'k3': distortion[4],
            'aperture_radius': 0.00268,
            'focus_distance': focus_distance / 100.0,

            'to_world': mi.ScalarTransform4f(cam_pose.tolist()),
            
            'film': {
                'type': 'hdrfilm',
                'width': render_size[1],
                'height': render_size[0],
                'pixel_format': 'rgb',
                'rfilter': {'type': 'gaussian'}
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            }
        },
        
        'calib_board': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([0.105, 0.1485, 1.0]), # todo: replace magic numbers with a proper expression
            'bsdf': {
                'type': 'twosided', 
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'bitmap',
                        'filename': calib_image_path,
                        'filter_type': 'bilinear',
                    }
                }
            },
        },
        
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0]  # Pure white light (R, G, B)
            }
        },
        
        'integrator': {
            'type': 'path',
            'max_depth': 8,
        },
    }
    return scene_dict

def main():
    # load parameters
    img_set = list(bokeh_img_sets.values())[0]
    
    plane_pose = np.array(img_set.get_pose()['plane_pose'])
    cam_pose = np.linalg.inv(plane_pose)
    # convert from opencv to mitsuba coords
    cam_pose = cam_pose @ np.diag([-1.0, -1.0, 1.0, 1.0])
    
    calib_img_path = img_set.get_gt_path()
    
    depth_min, depth_max = img_set.get_focus_distance_range(img_set.in_focus)
    depth = (depth_min + depth_max)*0.5
    
    calib = img_set.get_calib(img_set.in_focus)
    cam_mat = np.array(calib["camera_matrix"])
    
    photo = img_set.read_img(img_set.in_focus)/255

    # Create scene
    scene_dict = create_mitsuba_scene(cam_pose, calib_img_path, cam_mat, calib["distortion_coefficients"][0], depth, photo.shape[:2])
    scene = mi.load_dict(scene_dict)

    # Render
    print("Rendering scene...")
    image = mi.render(scene, spp=64)  # 64 samples per pixel
    
    # display result overlayed on gt scene
    plt.imshow((np.array(image) + photo)/2)
    plt.show()

    # Save result
    mi.util.write_bitmap('rendered_scene.png', image)
    print("Rendered image saved as rendered_scene.png")

if __name__ == "__main__":
    main()