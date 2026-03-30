import mitsuba as mi
import json
import numpy as np
import os
from calculate_fov import get_fov
from analysis import get_depth
from imgSet import bokeh_img_sets

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

def create_mitsuba_scene(matrix, calib_image_path, fov, focus_distance):
    """Create Mitsuba scene dictionary"""
    
    # 1. Convert OpenCV World-to-Camera to Camera-to-World
    T_cv_w2c = np.array(matrix).reshape(4, 4)
    T_cv_c2w = np.linalg.inv(T_cv_w2c)
    
    # 2. Fix the Coordinate System Flip (OpenCV -> Mitsuba)
    # Mitsuba's camera looks down +Z, Y-up, X-left. 
    # We flip the local X and Y axes to match.
    flip_xy = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_mi_c2w = T_cv_c2w @ flip_xy
    
    print(f"Final Mitsuba Camera-to-World Matrix:\n{T_mi_c2w}")
    
    scene_dict = {
        'type': 'scene',
        
        # Changed key from 'camera' to 'sensor' (Mitsuba standard)
        'sensor': {
            'type': 'thinlens',
            'fov': fov,
            'aperture_radius': 0.00268,
            'focus_distance': focus_distance / 100.0,
            
            # Apply the Extrinsics to the Camera, not the plane!
            'to_world': mi.ScalarTransform4f(T_mi_c2w.tolist()),
            
            'film': {
                'type': 'hdrfilm',
                'width': 8192,
                'height': 5464,
                'pixel_format': 'rgb',
                'rfilter': {'type': 'gaussian'} # Always good to explicitly define the filter
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            }
        },
        
        'calib_board': {
            'type': 'rectangle',
            # NO translation needed. The rectangle natively sits at (0,0,0)
            # We just scale it to the exact A4 physical dimensions
            'to_world': mi.ScalarTransform4f().scale([0.105, 0.1485, 1.0]),
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
    img_set = list(bokeh_img_sets.values())[0]
    
    pose = img_set.get_pose()
    pose_matrix = np.array(pose['transformation_matrix']).flatten().tolist()
    
    calib_img_path = img_set.get_gt_path()
    
    depth_min, depth_max = img_set.get_focus_distance_range(img_set.in_focus)
    depth = (depth_min + depth_max)*0.5

    # Create scene
    scene_dict = create_mitsuba_scene(pose_matrix, calib_img_path, get_fov(depth)[0], depth)
    scene = mi.load_dict(scene_dict)

    # Render
    print("Rendering scene...")
    image = mi.render(scene, spp=64)  # 128 samples per pixel

    # Save image
    mi.util.write_bitmap('rendered_scene.png', image)
    print("Rendered image saved as rendered_scene.png")

if __name__ == "__main__":
    main()