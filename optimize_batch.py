import json
import argparse
import cv2
import drjit as dr
import mitsuba as mi
import numpy as np
from imgSet import dot_stack_sets
import math
from common import *
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

def gen_mask(plane_pose, cam_mat, shape):
    """
    plane_pose: 4x4 matrix (Extrinsics)
    cam_mat: 3x3 matrix (Intrinsics)
    shape: (height, width) of the target output image
    paper_size_m: (width, height) of the physical paper in meters
    paper_size_px: (width, height) of the mask you are warping
    """
    
    S = np.array([
        [m_per_px, 0, -0.5*paper_size_m[0]],
        [0, m_per_px, -0.5*paper_size_m[1]],
        [0, 0, 1]
    ])

    H_pose = cam_mat @ plane_pose[:3, [0, 1, 3]]
    
    H = H_pose @ S

    base_mask = np.ones((int(paper_size_px[1]), int(paper_size_px[0])), dtype=np.uint8)

    mask_warped = cv2.warpPerspective(
        base_mask, 
        H, 
        (shape[1], shape[0]), 
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return mask_warped > 0

def get_crop_window_from_mask(mask, padding=50):
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("Mask is completely black!")

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    h, w = mask.shape
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(w, x_max + padding)
    y_end = min(h, y_max + padding)

    return (int(x_start), int(y_start), int(x_end - x_start), int(y_end - y_start))


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
                'pixel_format': 'luminance',
                'rfilter': {'type': 'gaussian'}
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 4
            }
        },
        'calib_board': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([paper_size_m[0]/2, paper_size_m[1]/2, 1.0]),
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


def optimize_single_target(img_set, target_idx, downscale=2):
    # 1. Coordinate Setup
    plane_pose = np.array(img_set.get_pose()['plane_pose'])
    cam_pose = np.linalg.inv(plane_pose) @ np.diag([-1.0, -1.0, 1.0, 1.0])
    
    # 2. Extract Depth and FOV baseline
    depth = img_set.get_focus_distance_range(target_idx)[2]
    cam_mat, dist_coeffs = load_calibration(checkerboard_single_path / "calib.json")
    cam_mat[:2] /= downscale
    
    # 3. Load ground truth image
    photo = img_set.read_img(target_idx).astype(np.float32)/255.0
    photo_grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    photo_small = cv2.resize(photo_grey, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_AREA)
    photo_lin = np.power(photo_small, 2.2)

    # 3.1 undistortion
    h, w = photo_small.shape[:2]
    perfect_cam_mat = cam_mat.copy()
    perfect_cam_mat[:2, 2] = (w/2, h/2)
    photo_aligned = cv2.undistort(photo_lin, cam_mat, dist_coeffs, None, perfect_cam_mat)
    
    fx_orig = float(cam_mat[0, 0])
    fov_x_deg = np.degrees(2.0 * np.arctan(w / (2.0 * fx_orig)))

    # 3.2 cropping
    mask = gen_mask(plane_pose, perfect_cam_mat, photo_small.shape)
    cx, cy, cw, ch = get_crop_window_from_mask(mask, padding=50)
    photo_crop = photo_aligned[cy:cy+ch, cx:cx+cw]
    
    # 3.3 normalization (put masked region in 0-1 range)
    low = np.min(photo_lin, initial=0, where=mask)
    high = np.max(photo_lin, initial=1, where=mask)
    photo_normalized = (photo_crop - low)/(high - low)
    
    ground = mi.TensorXf(np.ascontiguousarray(photo_normalized[:,:,None]))

    # 5. Scene Loading
    scene_dict = create_mitsuba_scene(cam_pose, img_set.get_pattern_path(), fov_x_deg, depth, 
                                     photo_aligned.shape[:2], cx, cy, cw, ch)
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    # 6. Optimized Parameters Setup
    epochs = 25
    lr_fov_min, lr_fov_max = 1e-2, 1e0
    lr_trans_min, lr_trans_max = 1e-8, 1e-5
    
    opt_fov = dr.opt.Adam(lr=lr_fov_max)
    opt_fov['x_fov'] = params['sensor.x_fov']
    
    opt_trans = dr.opt.Adam(lr=lr_trans_max)
    opt_trans['tx'], opt_trans['ty'] = mi.Float(0.0), mi.Float(0.0)
    
    base_board_transform = mi.Transform4f.scale([float(paper_size_m[0]/2), float(paper_size_m[1]/2), 1.0])

    print(f"Optimizing target_idx {target_idx} | Starting FOV: {fov_x_deg:.4f}°")

    # 7. Optimization Loop
    for i in range(epochs):
        # apply cosine annealing
        lr_fov = lr_fov_min + 0.5*(lr_fov_max - lr_fov_min)*(1 + math.cos(math.pi*i/epochs))
        opt_fov.set_learning_rate(lr_fov)
        
        lr_trans = lr_trans_min + 0.5*(lr_trans_max - lr_trans_min) * (1 + math.cos(math.pi*i/epochs))
        opt_trans.set_learning_rate(lr_trans)
        
        # Apply transforms
        offset = mi.Vector3f(opt_trans['tx'], opt_trans['ty'], mi.Float(0.0))
        params['calib_board.to_world'] = mi.Transform4f.translate(offset) @ base_board_transform
        params['sensor.x_fov'] = opt_fov['x_fov']
        params.update()
        
        # Render and take Loss
        render = mi.render(scene, params, seed=i, spp=4)
        loss = dr.mean(dr.square(render - ground))
        #plt.imshow(np.array(mi.Bitmap(dr.square(render - ground))))
        #plt.show()

        # Step
        dr.backward(loss)
        opt_fov.step()
        opt_trans.step()
        dr.eval(opt_fov['x_fov'], opt_trans['tx'], opt_trans['ty'])

        print(f"Epoch {i:02d} | Loss: {loss.array[0]:.5f} | FOV: {opt_fov['x_fov'][0]:.4f}° | dx: {opt_trans['tx'][0]:.5f}")

    # 8. Post-Optimization Physics Math
    fov_opt = float(opt_fov['x_fov'][0])
    tx_val, ty_val = float(opt_trans['tx'][0]), float(opt_trans['ty'][0])
    
    fx_new = (w / 2.0) / np.tan(np.radians(fov_opt) / 2.0)
    fy_new = fx_new * (cam_mat[1, 1] / cam_mat[0, 0]) # Keep aspect ratio
    
    # Convert physical shift (m) to pixel shift
    pixel_shift_x = fx_new * (tx_val / depth)
    pixel_shift_y = fy_new * (ty_val / depth)
    
    # Build Corrected Matrix
    new_cam_mat = np.array([
        [fx_new, 0, cam_mat[0, 2] - pixel_shift_x],
        [0, fy_new, cam_mat[1, 2] + pixel_shift_y],
        [0, 0, 1]
    ])
    new_cam_mat[:2] *= downscale

    # 9. JSON Output
    result_data = {
        "target_idx": int(target_idx),
        "depth_m": float(depth),
        "optimal_fov_deg": fov_opt,
        "tx_m": tx_val,
        "ty_m": ty_val,
        
        # Original matrix for comparison
        "original_camera_matrix": cam_mat.tolist(),
        "original_distortion": dist_coeffs.tolist(),
        
        # The corrected calibration for this target_idx
        "camera_matrix": new_cam_mat.tolist(),
        "distortion_coefficients": dist_coeffs.tolist()
    }
    return result_data

def main(resume=True):
    for set_id in dot_stack_sets.keys():
        img_set = dot_stack_sets[set_id]

        print(f"Running optimization for {set_id}")

        for target_idx in range(0, img_set.count):
            depth = img_set.get_focus_distance_range(target_idx)[2]
            output_path = optimized_calibration_path / f'calib_{set_id}_{depth}.json'
            if output_path.exists() and resume:
                continue
            
            print(f"\nProcessing index {target_idx} of {img_set.count}, depth: {depth}")
            
            result_data = optimize_single_target(img_set, target_idx)
            if result_data is None:
                print("Optimization failed, skipping")
                continue

            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            print(f"Saved optimized matrix for depth {depth} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis helper for camera calibration and depth plots.')
    
    main()
