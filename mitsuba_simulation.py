import mitsuba as mi
import json
import numpy as np
import os
from calculate_fov import get_fov
from analysis import get_depth
from imgSet import bokeh_img_sets

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

def load_first_pose(pose_file):
    """Load the first transformation matrix from pose_estimations.json"""
    with open(pose_file, 'r') as f:
        poses = json.load(f)

    first_key = next(iter(poses))
    matrix = poses[first_key]['transformation_matrix']
    return np.array(matrix).flatten().tolist(), first_key

def load_base_calib():
    """Loads the in-focus calibration for an image set"""

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
    pose_file = 'pose_estimations.json'
    calib_images_dir = 'calib_images'

    # Load first pose
    matrix, imgset_id = load_first_pose(pose_file)
    print(f"Using pose from {imgset_id}")

    # Find corresponding calib image
    calib_image_path = os.path.join(calib_images_dir, f'calib_{imgset_id}.png')
    if not os.path.exists(calib_image_path):
        print(f"Calib image not found: {calib_image_path}")
        return

    # Create scene
    scene_dict = create_mitsuba_scene(matrix, calib_image_path,get_fov(get_depth(bokeh_img_sets[imgset_id], 15))[0], get_depth(bokeh_img_sets[imgset_id], 15))
    scene = mi.load_dict(scene_dict)

    # Render
    print("Rendering scene...")
    image = mi.render(scene, spp=64)  # 64 samples per pixel

    # Save image
    mi.util.write_bitmap('rendered_scene.png', image)
    print("Rendered image saved as rendered_scene.png")

if __name__ == "__main__":
    main()