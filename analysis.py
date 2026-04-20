import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from imgSet import dot_stack_sets
import scipy.interpolate as inter
from common import *

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
    
    sorted_groups = sorted(depth_groups, key=lambda x: 1/x.depth)
    
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
    
    ax1.plot(depth, fx, marker='o', label='$f_x$')
    ax1.plot(depth, fy, marker='s', label='$f_y$')
    ax1.set_title('Focal Length ($f_x, f_y$) vs Inverse Depth')
    ax1.set_xlabel('Inverse Depth')
    ax1.set_ylabel('Pixels')
    ax1.legend()
    ax1.grid(True)

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
    
    axs[0, 0].plot(depth, k1, marker='o', color='blue')
    axs[0, 0].set_title('Radial Distortion $k_1$ vs Inverse Depth')
    
    axs[0, 1].plot(depth, k2, marker='s', color='orange')
    axs[0, 1].set_title('Radial Distortion $k_2$ vs Inverse Depth')

    axs[1, 0].plot(depth, k3, marker='^', color='green')
    axs[1, 0].set_title('Radial Distortion $k_3$ vs Inverse Depth')

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
    
"""
pattern results:
dots performed significantly better than hoops
tightly packed objects performed poorly

todo: run colmap for good patterns to decide which is the best overall

good:
or15_ir0_ds40
 - large dots
or10_ir0_ds30
 - med dots
or1_ir0_ds20
 - very small dots

mid:
or10_ir5_ds30
 - med hoops
or6_ir0_ds20
 - small dots
or15_ir7_ds40
 - large hoops

bad:
or10_ir1_ds20
 - tightly packed hoops
or10_ir2_ds20
 - tightly packed hoops
or6_ir3_ds20
 - small hoops
"""

def plot_focal_vs_depth(show_baseline, show_interpolated):
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # plot optimized calibrations for each pattern
    
    for imgset in dot_stack_sets.values():
        data = []
        for i in range(imgset.count):
            calib = imgset.get_optim_calib(i)
            if calib is None:
                continue
            K = calib[0]
            
            invdepth = 1/imgset.get_focus_distance_range(i)[2]
            fx = K[0,0]
            fy = K[1,1]
            data.append((invdepth,fx,fy))
        
        if data == []:
            continue
        data_sort = np.array(sorted(data))
        
        if show_interpolated:
            x = np.linspace(np.min(data_sort[:,0]), np.max(data_sort[:,0]), 300)
            fxs = np.zeros(len(x))
            fys = np.zeros(len(x))
            
            for i in range(len(x)):
                K = imgset.interpolate_calib(1/x[i])[0]
                fxs[i] = K[0,0]
                fys[i] = K[1,1]
        
            axs[0].plot(x, fxs, label=imgset.id)
            axs[1].plot(x, fys, label=imgset.id)
        else:
            axs[0].plot(data_sort[:,0], data_sort[:,1], label=imgset.id)
            axs[1].plot(data_sort[:,0], data_sort[:,2], label=imgset.id)
    
    # plot estimated calibrations from depth groups
    if show_baseline:
        from depthGroup import depth_groups
        
        sorted_groups = sorted(depth_groups, key=lambda x: 1/x.depth)
        
        invdepth = []
        fx, fy = [], []
        
        for group in sorted_groups:
            calib = group.read_calibration()
            if calib != {}:
                invdepth.append(1/group.depth)
                mat = np.array(calib["camera_matrix"])
                fx.append(mat[0,0])
                fy.append(mat[1,1])
                
        if show_interpolated:
            x = np.linspace(np.min(invdepth), np.max(invdepth), 300)
            
            cx = inter.CubicSpline(invdepth, fx, bc_type='natural')
            cy = inter.CubicSpline(invdepth, fy, bc_type='natural')
        
            axs[0].plot(x, cx(x), label="Checkerboard Stacks", color='black')
            axs[1].plot(x, cy(x), label="Checkerboard Stacks", color='black')
        else:
            axs[0].plot(invdepth, fx, label="Checkerboard Stacks", color='black')
            axs[1].plot(invdepth, fy, label="Checkerboard Stacks", color='black')
    
    axs[0].set_title('Focal Length $f_x$ vs Inverse Depth')
    axs[0].set_ylabel('$f_x$ (Pixels)')
    axs[0].set_xlabel('Inverse Depth ($\\text{Meters}^{-1}$)')
    axs[0].legend()
    axs[0].grid(True)
        
    axs[1].set_title('Focal Length $f_y$ vs Inverse Depth')
    axs[1].set_ylabel('$f_y$ (Pixels)')
    axs[1].set_xlabel('Inverse Depth ($\\text{Meters}^{-1}$)')
    axs[1].legend()
    axs[1].grid(True)
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis helper for camera calibration and depth plots.')
    parser.add_argument('--optim-results', action='store_true', help='Plot fx and fy vs depth from optimized_camera_matrix JSON files.')
    parser.add_argument('--optim-show-interpolated', action='store_true', help="Show interpolated optimized parameters")
    parser.add_argument('--optim-show-baseline', action='store_true', help="Show baseline parameters from checkerboard stacks")
    parser.add_argument('--focus-breathing', action='store_true', help='Show focus breathing plots.')
    args = parser.parse_args()

    if args.optim_results:
        plot_focal_vs_depth(show_baseline=args.optim_show_baseline, show_interpolated=args.optim_show_interpolated)

    if args.focus_breathing:
        disp_focus_breathing()

