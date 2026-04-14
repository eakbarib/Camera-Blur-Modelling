import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from imgSet import bokeh_img_sets

sheet_size = torch.tensor([0.210, 0.297])
sheet_pixels = torch.tensor([2048, 2896])
meters_per_sheet_pixel = sheet_size[0]/sheet_pixels[0]

def smoothstep(a, b, x):
    t = torch.clip((x - a)/(b - a), 0, 1)
    return t*t*(3 - 2*t)

def distort(img, distortion):
    pass

def get_screen_depth(pose, K, shape):
    n = pose[:3, 2]
    p = pose[:3, 3]
    d = torch.dot(n, p)

    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
    ph = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()

    K_inv = torch.inverse(K)
    rays = torch.einsum('ij, hwj -> hwi', K_inv, ph)

    q = torch.einsum('i, hwi -> hw', n, rays)
    depth = d/q

    return depth

def project_calib(plane_img, plane_pose, cam_mat, shape):
    H = cam_mat @ plane_pose[:3,[0,1,3]]
    H_inv = torch.inverse(H)
    
    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
    ph = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    
    qh = torch.einsum('ij, hwj -> hwi', H_inv, ph)
    q = qh[...,:2]/qh[...,2:]
    
    uv = 2*q/sheet_size
    
    img = F.grid_sample(plane_img[None,None,:,:,0].float(), uv[None,:,:,:], mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
    return img

def blur_curve(i):
    return curve_rate*(torch.exp(i/curve_rate) - 1)

def inv_blur_curve(k):
    return curve_rate*torch.log(k/curve_rate + 1)

def make_kernel_stack(radi):
    l = 2*torch.ceil(torch.max(radi)).int() + 1
    t = torch.arange(l) + 0.5 - l/2
    uv = torch.stack(torch.meshgrid([t, t], indexing="ij"), dim=2)
    d = torch.linalg.norm(uv, dim=2)
    kernels = smoothstep(-0.5, 0.5, radi[:,None,None] - d[None,:,:])
    return (l, kernels.reshape(radi.shape[0], l*l))

def apply_blur(img, blurmap):
    kernel_size, kernels = make_kernel_stack(blurmap.flatten()).unsqueeze(0)
    img_cols_ref = torch.nn.unfold(img.unsqueeze(0), kernel_size, padding=kernel_size//2)
    img_cols = img_cols_ref.clone()
    conv = torch.sum(img_cols * kernels, dim=1)
    blurred = conv.reshape(1, 1, img.shape[1], img.shape[2]).squeeze(0)
    return blurred



img_set = list(bokeh_img_sets.values())[0]
plane_img = torch.from_numpy(img_set.read_gt())
in_focus_img = torch.from_numpy(img_set.read_img(img_set.in_focus))
res_shape = in_focus_img.shape

curve_rate = 10
max_blur_idx = 31
kernel_radi = blur_curve(torch.arange(max_blur_idx))

blur_kernels = []
for k in kernel_radi:
    l = 2*torch.ceil(k).int() + 1
    t = torch.arange(l)
    uv = torch.stack(torch.meshgrid([t, t], indexing="ij"), dim=2) + torch.full((1,1,2), 0.5 - l/2)
    d = torch.linalg.norm(uv, dim=2) - k
    kernel = 1 - smoothstep(-0.5, 0.5, d)
    blur_kernels.append(kernel[None,...])
    
#plt.imshow(blur_kernels[150])
#plt.show()
#plt.imshow(blur_kernels[151])
#plt.show()

plane_pose = torch.tensor(img_set.get_pose()['plane_pose'])

calib = img_set.get_calib(img_set.in_focus)
cam_mat = torch.tensor(calib["camera_matrix"])

#print(cam_pose)

#H = cam_mat @ torch.linalg.inv(plane_pose[:3,(0,1,3)])
depthmap = get_screen_depth(plane_pose, cam_mat, res_shape)

depth_min, depth_max = img_set.get_focus_distance_range(img_set.in_focus)
focus_dist = 0.1*(depth_min + depth_max)*0.5
focal_len_px = cam_mat[0,0]
px_to_meters = 0.015/focal_len_px # rough estimate
focal_len = focal_len_px*px_to_meters
aperture_radius = 0.015/(2*2.8) # from lens specs

delta_focus = torch.abs(depthmap - focus_dist)
blurmap = focal_len_px*aperture_radius*delta_focus/(depthmap*(focus_dist - focal_len))
blurmap = torch.clamp(blurmap, 0, 10)

warped = project_calib(plane_img, plane_pose, cam_mat, res_shape)

blurred = apply_blur(warped, blurmap)

plt.imshow(warped)
plt.show()