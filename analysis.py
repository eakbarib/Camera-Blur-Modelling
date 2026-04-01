from imgSet import bokeh_img_sets
from depthGroup import depth_groups
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import scipy.optimize as opt

def disp_slices(slices):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    init_idx = slices.shape[0] // 2
    im = ax.imshow(slices[init_idx], cmap='gray')
    ax.set_title(f"Slice {init_idx}")

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])

    slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=slices.shape[0] - 1,
        valinit=init_idx,
        valstep=1
    )

    def update(val):
        index = int(slider.val)
        im.set_data(slices[index])
        ax.set_title(f"Slice {index}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
    
def focus_dist_curve(x, a, b, k):
    return k/(x - b) + a
    
def disp_focus_depth(img_set):
    fmin = np.empty(img_set.count)
    fmax = np.empty(img_set.count)
    for i in range(img_set.count):
        meta = img_set.read_meta(i)
        fmin[i] = meta["Exif.CanonFi.FocusDistanceLower"].getValue().toFloat()
        fmax[i] = meta["Exif.CanonFi.FocusDistanceUpper"].getValue().toFloat()
    
    x = np.arange(img_set.count)
    min_params, _ = opt.curve_fit(focus_dist_curve, x[:63], fmin[:63], p0=[0, 63, -1])
    max_params, _ = opt.curve_fit(focus_dist_curve, x[:63], fmax[:63], p0=[0, 63, -1])
    plt.plot(x, fmin)
    plt.plot(x, focus_dist_curve(x, *min_params))
    plt.plot(x, fmax)
    plt.plot(x, focus_dist_curve(x, *max_params))
    plt.show()
    
def disp_focus_breathing():
    sorted_groups = sorted(depth_groups, key=lambda x: x.depth)
    
    depth = []
    fx, fy, cx, cy = [], [], [], []
    k1, k2, k3, p1, p2 = [], [], [], [], []
    
    for group in sorted_groups:
        calib = group.calibration
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

if __name__ == "__main__":
    # view focus breathing
    if True:
        disp_focus_breathing()
    
    # view focus derivative
    if False:
        sobel3d = [
            [[ 1, 2, 1],
            [ 2, 4, 2],
            [ 1, 2, 1]],
            [[ 0, 0, 0],
            [ 0, 0, 0],
            [ 0, 0, 0]],
            [[-1,-2,-1],
            [-2,-4,-2],
            [-1,-2,-1]]
        ]
        
        stack = bokeh_img_sets["or6_ir0_ds20"].get_stack()
        derivative_focus = convolve(np.float32(stack), sobel3d, mode='constant', cval=0.0, axes=(0,1,2))
        disp_slices(derivative_focus)

    # view depth curves
    if False:
        disp_focus_depth(bokeh_img_sets["or6_ir0_ds20"])

        for img_set in bokeh_img_sets.values():
            fmin, fmax = get_depths(img_set)
            plt.plot(fmin)
            plt.show()

