import cv2

# Load the original digital PNG
img = cv2.imread('calib_or6_ir3_ds20.png')

# Get the total pixel width of the canvas
pixel_width = img.shape[1]

# A4 paper width in meters
a4_width_m = 0.210 

meters_per_pixel = a4_width_m / pixel_width
print(f"Canvas Pixel Width: {pixel_width} px")
print(f"Meters per Pixel: {meters_per_pixel:.8f}")