import json
import os
import gc
import cv2
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
from imgSet import bokeh_img_sets

mi.set_variant('cuda_ad_rgb')


def get_crop_window_from_mask(mask_path, padding=50):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask at {mask_path}")

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("Mask is completely black!")

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    h, w = mask.shape
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(w, x_max + padding)
    y_end = min(h, y_max + padding)

    crop_width = x_end - x_start
    crop_height = y_end - y_start
    crop_window = [int(x_start), int(y_start), int(crop_width), int(crop_height)]
    return crop_window, (int(y_start), int(y_end), int(x_start), int(x_end))


def create_mitsuba_scene(cam_pose, calib_image_path, fov_x_deg, focus_distance, render_size, x_offset, y_offset, crop_w, crop_h):
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
                'sample_count': 4
            }
        },
        'calib_board': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([0.105, 0.1485, 1.0]),
            'bsdf': {
                'type': 'twosided',
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'bitmap',
                        'filename': calib_image_path,
                        'filter_type': 'bilinear'
                    }
                }
            }
        },
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0]
            }
        },
        'integrator': {
            'type': 'path',
            'max_depth': 2
        }
    }
    return scene_dict


def optimize_single_target(img_set, target_idx, output_dir):
    # 1. Coordinate Setup
    plane_pose = np.array(img_set.get_pose()['plane_pose'])
    cam_pose = np.linalg.inv(plane_pose) @ np.diag([-1.0, -1.0, 1.0, 1.0])
    
    # 2. Extract Depth and FOV baseline
    depth_min, depth_max = img_set.get_focus_distance_range(target_idx)
    depth = (depth_min + depth_max) * 0.5
    calib = img_set.get_calib(img_set.in_focus)
    cam_mat = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["distortion_coefficients"][0])
    photo = img_set.read_img(target_idx) / 255.0

    calib_gt = img_set.get_calib(target_idx)
    if calib_gt is None:
        print(f"No calibration found for target_idx {target_idx}; skipping output JSON")
        return None
    gt_cam_mat = np.array(calib_gt["camera_matrix"])

    # 3. Aligned Undistortion
    h, w = photo.shape[:2]
    perfect_cam_mat = cam_mat.copy()
    perfect_cam_mat[0, 2], perfect_cam_mat[1, 2] = w / 2.0, h / 2.0
    photo_aligned = cv2.undistort(photo, cam_mat, dist_coeffs, None, perfect_cam_mat)
    
    fx_orig = float(cam_mat[0, 0])
    fov_x_deg = np.degrees(2.0 * np.arctan(w / (2.0 * fx_orig)))

    # 4. Prepare Cropped Ground Truth (NumPy-side for VRAM safety)
    crop_list, (y1, y2, x1, x2) = get_crop_window_from_mask("./boardmask.png", padding=50)
    photo_crop = photo_aligned[y1:y2, x1:x2]
    photo_crop_linear = np.power(photo_crop, 2.2).astype(np.float32)
    gt_crop = mi.TensorXf(np.ascontiguousarray(photo_crop_linear))

    # 5. Scene Loading
    scene_dict = create_mitsuba_scene(cam_pose, img_set.get_gt_path(), fov_x_deg, depth, 
                                     photo_aligned.shape[:2], x1, y1, crop_list[2], crop_list[3])
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    # 6. Optimized Parameters Setup
    opt_fov = dr.opt.Adam(lr=0.1)
    opt_fov['x_fov'] = params['sensor.x_fov']
    
    opt_trans = dr.opt.Adam(lr=0.001)
    opt_trans['tx'], opt_trans['ty'] = mi.Float(0.0), mi.Float(0.0)
    
    base_board_transform = mi.Transform4f.scale([0.105, 0.1485, 1.0])

    print(f"Optimizing target_idx {target_idx} | Starting FOV: {fov_x_deg:.4f}°")

    # 7. The Loop
    for i in range(25):
        # Apply transforms
        offset = mi.Vector3f(opt_trans['tx'], opt_trans['ty'], mi.Float(0.0))
        params['calib_board.to_world'] = mi.Transform4f.translate(offset) @ base_board_transform
        params['sensor.x_fov'] = opt_fov['x_fov']
        params.update()

        # Render and Loss
        image_crop = mi.render(scene, params, seed=i, spp=4)
        loss = dr.mean(dr.square(image_crop - gt_crop))

        dr.backward(loss)
        opt_fov.step()
        opt_trans.step()
        dr.eval(opt_fov['x_fov'], opt_trans['tx'], opt_trans['ty'])

        print(f"Epoch {i:02d} | Loss: {loss.array[0]:.5f} | FOV: {opt_fov['x_fov'][0]:.4f}° | dx: {opt_trans['tx'][0]:.5f}")

        # The Nuclear Memory Reset
        del image_crop, loss
        gc.collect()

    # 8. Post-Optimization Physics Math
    fov_opt = float(opt_fov['x_fov'][0])
    tx_val, ty_val = float(opt_trans['tx'][0]), float(opt_trans['ty'][0])
    
    fx_new = (w / 2.0) / np.tan(np.radians(fov_opt) / 2.0)
    fy_new = fx_new * (cam_mat[1, 1] / cam_mat[0, 0]) # Keep aspect ratio
    
    # Convert physical shift (m) to pixel shift
    pixel_shift_x = fx_new * (tx_val / depth)
    pixel_shift_y = fy_new * (ty_val / depth)
    
    # Build Corrected Matrix
    new_cam_mat = cam_mat.copy()
    new_cam_mat[0, 0], new_cam_mat[1, 1] = fx_new, fy_new
    new_cam_mat[0, 2] = cam_mat[0, 2] - pixel_shift_x
    new_cam_mat[1, 2] = cam_mat[1, 2] + pixel_shift_y

    # 9. JSON Output
    result_data = {
        "target_idx": int(target_idx),
        "depth_m": float(depth),
        "optimal_fov_deg": fov_opt,
        "tx_m": tx_val,
        "ty_m": ty_val,
        
        # Original matrix for comparison
        "original_camera_matrix": cam_mat.tolist(),
        
        # The specific corrected matrix for this target_idx
        "camera_matrix": new_cam_mat.tolist(),
        
        "gt_camera_matrix": gt_cam_mat.tolist()
    }
    return result_data

def main():
    img_set = list(bokeh_img_sets.values())[0]
    start_idx = 27
    end_idx = min(img_set.count - 1, img_set.in_focus + 14)

    print(f"Running optimization for indices {start_idx} through {end_idx} (inclusive)")

    output_dir = 'optimized_camera_matrix'
    os.makedirs(output_dir, exist_ok=True)

    for target_idx in range(start_idx, end_idx + 1):
        print(f"\nProcessing target_idx {target_idx}")
        result_data = optimize_single_target(img_set, target_idx, output_dir)
        if result_data is None:
            continue

        output_path = os.path.join(output_dir, f'optimized_matrix_idx_{target_idx}.json')
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"Saved optimized matrix for target_idx {target_idx} to {output_path}")


if __name__ == "__main__":
    main()
