import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')

import numpy as np
from imgSet import bokeh_img_sets
import matplotlib.pyplot as plt
import cv2

import cv2
import numpy as np

def get_crop_window_from_mask(mask_path, padding=50):
    # 1. Load the mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Could not load mask at {mask_path}")

    # 2. Find the coordinates of all white pixels (values > 0)
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("Mask is completely black!")

    # 3. Find the extreme edges
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    # 4. Add padding, ensuring we don't go outside the image bounds
    h, w = mask.shape
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(w, x_max + padding)
    y_end = min(h, y_max + padding)

    # 5. Calculate final width and height for Mitsuba
    crop_width = x_end - x_start
    crop_height = y_end - y_start

    # Mitsuba format: [x_offset, y_offset, width, height]
    crop_window = [int(x_start), int(y_start), int(crop_width), int(crop_height)]
    
    # We also return the slicing indices for the Ground Truth image!
    return crop_window, (int(y_start), int(y_end), int(x_start), int(x_end))

def create_mitsuba_scene(cam_pose, calib_image_path, fov_x_deg, focus_distance, render_size, x_offset, y_offset, crop_w, crop_h):
    """Create Mitsuba scene dictionary"""
    
    scene_dict = {
        'type': 'scene',
        
        'sensor': {
            'type': 'thinlens',
            'aperture_radius': 0.00268,
            'focus_distance': focus_distance / 100.0,

            'to_world': mi.ScalarTransform4f(cam_pose.tolist()),
            'fov': fov_x_deg,
            'fov_axis': 'x',

            'film': {
                'type': 'hdrfilm',
                'width': render_size[1],
                'height': render_size[0],
                'crop_offset_x': int(x_offset),
                'crop_offset_y': int(y_offset),
                'crop_width': int(crop_w),
                'crop_height': int(crop_h),
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
            'to_world': mi.ScalarTransform4f().scale([0.105, 0.1485, 1.0]), # todo: replace magic numbers
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
            'max_depth': 2,
        },
    }
    return scene_dict

def main():
    # 1. Define the target frame (the blurry one)
    img_set = list(bokeh_img_sets.values())[0]
    target_idx = img_set.in_focus + 10

    # Load pose (Assuming the camera didn't physically move, pose is static)
    plane_pose = np.array(img_set.get_pose()['plane_pose'])
    cam_pose = np.linalg.inv(plane_pose)
    cam_pose = cam_pose @ np.diag([-1.0, -1.0, 1.0, 1.0])

    calib_img_path = img_set.get_gt_path()

    depth_min, depth_max = img_set.get_focus_distance_range(img_set.in_focus)
    depth = (depth_min + depth_max) * 0.5

    # Fetch calibration and photo for the blurry frame
    calib = img_set.get_calib(target_idx)
    cam_mat = np.array(calib["camera_matrix"])
    distortion_coeffs = np.array(calib["distortion_coefficients"][0])
    photo = img_set.read_img(target_idx) / 255.0

    # --- PRINCIPAL POINT SHIFT & UNDISTORTION ---
    h, w = photo.shape[:2]

    perfect_cam_mat = cam_mat.copy()
    perfect_cam_mat[0, 2] = w / 2.0  # Force exact center X
    perfect_cam_mat[1, 2] = h / 2.0  # Force exact center Y

    photo_aligned = cv2.undistort(photo, cam_mat, distortion_coeffs, None, perfect_cam_mat)
    # ---------------------------------------------

    # --- FOV CALCULATION (Initial Guess) ---
    fx = float(cam_mat[0, 0])
    fov_x_rad = 2.0 * np.arctan(w / (2.0 * fx))
    fov_x_deg = np.degrees(fov_x_rad)
    # ---------------------------------------------

    mask_file = "./boardmask.png"  # Binary mask where the ArUco board is white and the rest is black
    
    # 1. Call our helper function
    crop_list, (y1, y2, x1, x2) = get_crop_window_from_mask(mask_file, padding=50)
    
    # 2. Unpack the list for the scene creator
    _, _, crop_w, crop_h = crop_list
    # ---------------------------------------------

    # Create scene (Make sure you are using the 'diff_thinlens' plugin!)
    scene_dict = create_mitsuba_scene(
        cam_pose, calib_img_path, fov_x_deg,
         depth, photo_aligned.shape[:2],
        x1, y1, crop_w, crop_h
    )
    scene = mi.load_dict(scene_dict)

    # Prepare Ground Truth for Optimizer (Linearize and Crop)
    photo_linear = np.power(photo_aligned, 2.2)
    gt_image = mi.TensorXf(photo_linear)
    gt_crop = gt_image[y1:y2, x1:x2]

    # Setup Optimizer for FOV (Focus Breathing compensation)
    params = mi.traverse(scene)
    opt = dr.opt.Adam(lr=0.5)
    opt['x_fov'] = params['sensor.x_fov']

    print(f"Starting FOV (from calib): {opt['x_fov'][0]:.4f} degrees")

    # The Auto-Alignment Loop
    epochs = 25
    for i in range(epochs):
        params['sensor.x_fov'] = opt['x_fov']
        params.update()

        # Render (Outputs a tiny image because of crop_window, saving VRAM!)
        image_crop = mi.render(scene, params, seed=i, spp=4)

        # Calculate Loss ONLY on the ArUco board
        loss = dr.mean(dr.square(image_crop - gt_crop))

        # Backpropagate and step
        dr.backward(loss)
        opt.step()
        dr.eval(opt['x_fov'])

        print(f"Epoch {i:02d} | Loss: {loss.array[0]:.5f} | Learned FOV: {opt['x_fov'][0]:.4f}°")

    # --- FINAL FULL-RES RENDER ---
    print("\nOptimization complete. Rendering final full-resolution image...")

    # Remove the crop window to render the whole 8K frame safely
    params['sensor.film.crop_offset'] = [0, 0]
    params['sensor.film.crop_size'] = [int(w), int(h)]
    params.update()

    final_image = mi.render(scene, params, spp=64)

    # Display result overlayed on the aligned real scene
    plt.imshow((np.array(final_image) + photo_aligned) / 2.0)
    plt.title(f"Optimized FOV: {opt['x_fov'][0]:.4f}°")
    plt.show()

    # Save result
    mi.util.write_bitmap('rendered_scene.png', final_image)
    print("Rendered image saved as rendered_scene.png")


if __name__ == "__main__":
    main()