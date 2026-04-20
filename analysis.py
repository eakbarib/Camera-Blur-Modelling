import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
    
def focus_dist_curve(x, a, b, k):
    return k/(x - b) + a
    
def disp_focus_depth(img_set):
    fmin = np.empty(img_set.count)
    fmax = np.empty(img_set.count)
    for i in range(img_set.count):
        fmin[i], fmax[i], _ = img_set.get_focus_distance_range(i)
    
    x = np.arange(img_set.count)
    min_params, _ = opt.curve_fit(focus_dist_curve, x[:63], fmin[:63], p0=[0, 63, -1])
    max_params, _ = opt.curve_fit(focus_dist_curve, x[:63], fmax[:63], p0=[0, 63, -1])
    plt.plot(x, fmin)
    plt.plot(x, focus_dist_curve(x, *min_params))
    plt.plot(x, fmax)
    plt.plot(x, focus_dist_curve(x, *max_params))
    plt.show()
    
def disp_focus_breathing():
    from depthGroup import depth_groups
    
    sorted_groups = sorted(depth_groups, key=lambda x: x.depth)
    
    depth = []
    fx, fy, cx, cy = [], [], [], []
    k1, k2, k3, p1, p2 = [], [], [], [], []
    
    for group in sorted_groups:
        calib = group.read_calibration()
        if calib != {}:
            depth.append(1/group.depth)
            mat = np.array(calib["camera_matrix"])
            fx.append(mat[0,0])
            fy.append(mat[1,1])
            cx.append(mat[0,2])
            cy.append(mat[1,2])
            dist = calib["distortion_coefficients"][0]
            k1.append(dist[0])
            k2.append(dist[1])
            p1.append(dist[2])
            p2.append(dist[3])
            k3.append(dist[4])
    
    # Figure 1
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # fx, fy vs depth
    ax1.plot(depth, fx, marker='o', label='$f_x$')
    ax1.plot(depth, fy, marker='s', label='$f_y$')
    ax1.set_title('Focal Length ($f_x, f_y$) vs Inverse Depth')
    ax1.set_xlabel('Inverse Depth')
    ax1.set_ylabel('Pixels')
    ax1.legend()
    ax1.grid(True)

    # cx, cy vs depth
    ax2.plot(depth, cx, marker='o', label='$c_x$')
    ax2.plot(depth, cy, marker='s', label='$c_y$')
    ax2.set_title('Principal Point ($c_x, c_y$) vs Inverse Depth')
    ax2.set_xlabel('Inverse Depth')
    ax2.set_ylabel('Pixels')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Figure 2 
    fig2, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # k1 vs depth
    axs[0, 0].plot(depth, k1, marker='o', color='blue')
    axs[0, 0].set_title('Radial Distortion $k_1$ vs Inverse Depth')
    
    # k2 vs depth
    axs[0, 1].plot(depth, k2, marker='s', color='orange')
    axs[0, 1].set_title('Radial Distortion $k_2$ vs Inverse Depth')

    # k3 vs depth
    axs[1, 0].plot(depth, k3, marker='^', color='green')
    axs[1, 0].set_title('Radial Distortion $k_3$ vs Inverse Depth')

    # p1, p2 vs depth
    axs[1, 1].plot(depth, p1, marker='d', label='$p_1$')
    axs[1, 1].plot(depth, p2, marker='x', label='$p_2$')
    axs[1, 1].set_title('Tangential Distortion $p_1, p_2$ vs Inverse Depth')
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xlabel('Inverse Depth')
        ax.set_ylabel('Coefficient Value')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def load_optimized_focal_data(opt_dir="optimized_camera_matrix"):
    records = []
    if not os.path.isdir(opt_dir):
        raise FileNotFoundError(f"Optimized camera matrix directory not found: {opt_dir}")

    for filename in os.listdir(opt_dir):
        if not filename.lower().endswith('.json'):
            continue
        path = os.path.join(opt_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        depth = data.get('depth_m')
        if depth is None:
            continue

        camera_matrix = np.array(data.get('camera_matrix', []), dtype=float)
        gt_camera_matrix = np.array(data.get('gt_camera_matrix', []), dtype=float)
        if camera_matrix.shape != (3, 3) or gt_camera_matrix.shape != (3, 3):
            continue

        records.append({
            'depth': depth,
            'fx': camera_matrix[0, 0],
            'fy': camera_matrix[1, 1],
            'gt_fx': gt_camera_matrix[0, 0],
            'gt_fy': gt_camera_matrix[1, 1],
        })

    records.sort(key=lambda x: x['depth'])
    return records


def plot_focal_vs_depth(opt_dir='optimized_camera_matrix'):
    records = load_optimized_focal_data(opt_dir)
    if not records:
        raise ValueError(f"No optimized camera matrix records found in {opt_dir}")

    depths = [r['depth'] for r in records]
    fx = [r['fx'] for r in records]
    fy = [r['fy'] for r in records]
    gt_fx = [r['gt_fx'] for r in records]
    gt_fy = [r['gt_fy'] for r in records]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(depths, fx, marker='o', label='camera_matrix $f_x$')
    axs[0].plot(depths, gt_fx, marker='s', label='gt_camera_matrix $f_x$')
    axs[0].set_title('Focal Length $f_x$ vs Depth')
    axs[0].set_ylabel('Pixels')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(depths, fy, marker='o', label='camera_matrix $f_y$')
    axs[1].plot(depths, gt_fy, marker='s', label='gt_camera_matrix $f_y$')
    axs[1].set_title('Focal Length $f_y$ vs Depth')
    axs[1].set_xlabel('Depth (m)')
    axs[1].set_ylabel('Pixels')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis helper for camera calibration and depth plots.')
    parser.add_argument('--plot-focal-vs-depth', action='store_true', help='Plot fx and fy vs depth from optimized_camera_matrix JSON files.')
    parser.add_argument('--optimized-camera-dir', default='optimized_camera_matrix', help='Directory containing optimized camera matrix JSON files.')
    parser.add_argument('--focus-breathing', action='store_true', help='Show focus breathing plots.')
    parser.add_argument('--no-default', action='store_true', help='Do not show default focus breathing plot when no other option is provided.')
    args = parser.parse_args()

    if args.plot_focal_vs_depth:
        plot_focal_vs_depth(args.optimized_camera_dir)

    if args.focus_breathing:
        disp_focus_breathing()

    if not args.plot_focal_vs_depth and not args.focus_breathing and not args.no_default:
        disp_focus_breathing()

