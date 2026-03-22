from imgSet import bokeh_img_sets
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

def get_depths(img_set):
    fmin = np.empty(img_set.count)
    fmax = np.empty(img_set.count)
    for i in range(img_set.count):
        meta = img_set.read_meta(i)
        fmin[i] = meta["Exif.CanonFi.FocusDistanceLower"].getValue().toFloat()
        fmax[i] = meta["Exif.CanonFi.FocusDistanceUpper"].getValue().toFloat()
    return (fmin, fmax)
    
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

if __name__ == "__main__":

    disp_focus_depth(bokeh_img_sets["or6_ir0_ds20"])

    for img_set in bokeh_img_sets.values():
        fmin, fmax = get_depths(img_set)
        plt.plot(fmin)
        plt.show()

#stack = img_sets["or6_ir0_ds20"].get_stack()
#derivative_focus = convolve(np.float32(stack), sobel3d, mode='constant', cval=0.0, axes=(0,1,2))
#disp_slices(derivative_focus)