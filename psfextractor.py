import cv2
import numpy as np
import rawpy
import os

def extract_dot_centroids(image_path, output_debug_image="detected_dots.png"):
    # 1. Load the sharp image in grayscale
    # Assuming you already converted the .CR3 to a .png or .jpg
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=1.0)
    gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # 2. Set up the Blob Detector Parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Color (0 looks for black blobs on a lighter background)
    params.filterByColor = True
    params.blobColor = 0

    # --- NEW: Fix for Dark / Linear Images ---
    params.minThreshold = 5      # Start looking at very dark pixel values
    params.maxThreshold = 200    
    params.thresholdStep = 10

    # Filter by Area (Adjust these based on your image resolution)
    # This ignores tiny dust specks and massive shadows
    params.filterByArea = True
    params.minArea = 2000      # Min pixels 
    params.maxArea = 15000    # Max pixels

    # Filter by Circularity (CRITICAL STEP)
    # A perfect circle has a circularity of 1.0. A square is roughly 0.78.
    # Setting the minimum to 0.85 ensures we ignore the ArUco markers!
    params.filterByCircularity = True
    params.minCircularity = 0.85

    # Filter by Convexity and Inertia (Ensures shapes are solid and not lines)
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # 3. Create the detector and find the blobs
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)

    # 4. Extract the (x, y) coordinates
    centroids = []
    for kp in keypoints:
        x, y = kp.pt
        centroids.append((int(x), int(y)))
        
    print(f"Successfully detected {len(centroids)} circular dots.")

    # 5. Visual Debugging (Optional but highly recommended)
    # This draws red circles around everything it found so you can verify
    # it didn't grab an ArUco marker by mistake.
    img_with_keypoints = cv2.drawKeypoints(
        gray_img, keypoints, np.array([]), 
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(output_debug_image, img_with_keypoints)
    print(f"Saved visual verification to: {output_debug_image}")

    return centroids

def extract_and_process_psfs(blurry_image_path, centroids, patch_size=100, output_dir="extracted_psfs"):
    # 1. Load the blurry image in grayscale
    # Make sure this is your linear RGB PNG, not the CR3
    if blurry_image_path.endswith(".CR3"):
        with rawpy.imread(blurry_image_path) as raw:
            blurry_img = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=1.0)
        blurry_img = cv2.cvtColor(blurry_img, cv2.COLOR_RGB2GRAY| cv2.IMREAD_ANYDEPTH)
    else:
        blurry_img = cv2.imread(blurry_image_path, cv2.IMREAD_GRAYSCALE| cv2.IMREAD_ANYDEPTH)
    
    if blurry_img is None:
        raise FileNotFoundError(f"Could not load blurry image: {blurry_image_path}")

    # Create a folder to save the extracted PSFs so you can look at them
    os.makedirs(output_dir, exist_ok=True)
    
    half_size = patch_size // 2
    processed_psfs = []

    print(f"Extracting PSFs for {len(centroids)} detected dots...")

    for i, (x, y) in enumerate(centroids):
        # 2. Boundary Checks
        # Ensure our crop box doesn't fall off the edge of the image
        y_start = y - half_size
        y_end = y + half_size
        x_start = x - half_size
        x_end = x + half_size

        if y_start < 0 or y_end > blurry_img.shape[0] or x_start < 0 or x_end > blurry_img.shape[1]:
            print(f"Skipping dot at ({x}, {y}) - too close to the image edge.")
            continue

        # 3. Crop the patch
        patch = blurry_img[y_start:y_end, x_start:x_end]

        # 4. Math Conversion (Black Dot to PSF Kernel)
        # We MUST convert to float32. If we stay in uint8, subtracting 
        # a dark pixel from a light pixel might underflow and wrap around to 255!
        patch_float = patch.astype(np.float32)
        
        # --- NEW: Denoise the raw sensor data ---
        # A 5x5 Gaussian blur eliminates the sharp grain while preserving the soft PSF
        patch_float = cv2.GaussianBlur(patch_float, (5, 5), 0)
        
        bg_estimate = np.percentile(patch_float, 90)  # Estimate the local background using the 90th percentile
        
        # 2. Adaptive Inversion
        # Instead of subtracting a single number, we subtract the local background gradient
        psf_raw = np.clip(bg_estimate - patch_float, 0, None)
        
        # 3. Clean up the residual noise
        # Now that the gradient is gone, a 15% floor will cleanly slice off the remaining haze
        noise_floor = np.max(psf_raw) * 0.15
        psf_raw[psf_raw < noise_floor] = 0.0
        
        
        # Normalize the kernel so all the light energy adds up to exactly 1.0
        sum_energy = np.sum(psf_raw)
        if sum_energy > 0:
            psf_normalized = psf_raw / sum_energy
        else:
            psf_normalized = psf_raw

        # Store the mathematical kernel and its original location
        processed_psfs.append({
            'x': x,
            'y': y,
            'kernel_matrix': psf_normalized
        })

        # 5. Visual Debugging
        # We can't save a normalized matrix (where values are like 0.0001) directly to a PNG.
        # We scale it back up to 0-255 just for the sake of looking at it in your file explorer.
        psf_vis = (psf_normalized / np.max(psf_normalized) * 255).astype(np.uint8)
        
        # Save the image patch
        cv2.imwrite(os.path.join(output_dir, f"psf_{i:03d}_x{x}_y{y}.png"), psf_vis)

    print(f"Done! Successfully extracted {len(processed_psfs)} valid PSFs.")
    print(f"Check the '{output_dir}' folder to see the aberration shapes.")
    
    return processed_psfs

# --- Example Usage ---
if __name__ == "__main__":
    sharp_image_file = "bokeh_calib_photos\\or6_ir0_ds20\\IMG_9626.CR3"
    blurry_image_file = "bokeh_calib_photos\\or6_ir0_ds20\\IMG_9621.CR3"
    
    # Fake coordinates for demonstration if you just want to test the crop logic
    # dot_coordinates = [(1000, 1000), (2000, 1500), (3500, 2500)]
    
    # Pass your highly defocused image here
    psf_data = extract_and_process_psfs(blurry_image_file, extract_dot_centroids(sharp_image_file),patch_size=200)
    pass