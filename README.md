# Camera-Blur-Modelling

Calibrating camera + lens depth of field blur using Mitsuba renderer for simulation.

## Project Files Description

### Core Scripts

- **`analysis.py`**: Contains functions for analyzing image sets and focus depths.
  - `disp_slices(slices)`: Displays slices of a 3D array using matplotlib with a slider.
  - `disp_focus_depth(img_set)`: Plots focus depth curves for an image set.
  - `def disp_focus_breathing()`: Plots lens parameters vs focus distance.

- **`calib_images.py`**: Generates calibration images with ArUco markers and dot patterns.
  - `gen_calib_image(true_marker_size, true_dot_outer_radius, true_dot_inner_radius, true_dot_spacing)`: Creates and saves calibration images with markers and dots for pose estimation.

- **`calibrate_depths.py`**: Performs camera calibration using checkerboard images grouped by depth.
  - `calibrate_camera(images)`: Calibrates camera intrinsics using OpenCV's checkerboard detection.
  - Main script processes depth folders and saves calibration data to JSON files.

- **`Camera.py`**: Defines a custom Mistuba sensor with differentible lens parameters

- **`depthGroup.py`**: Defines the `depthGroup` class for handling checkerboard depth groups.
  - `depthGroup.__init__(folder)`: Initializes a depth group from its folder
  - `read_img(idx)`: Returns an image from the depth group
  - Defines the `depth_groups` list.

- **`group_checkerboard_by_depth.py`**: Groups checkerboard images by their focus depth.
  - Organizes images from `checkerboard_img_sets` into depth-based folders for calibration.

- **`imgSet.py`**: Defines the `imgSet` class for handling image sets and related operations.
  - `imgSet.__init__(folder, set_id, start, in_focus, end)`: Initializes an image set with folder path and image range.
  - `imgSet.read_meta(idx)`: Reads EXIF metadata from a CR3 raw image file.
  - `imgSet.get_focus_distance_range(idx)`: Returns the range of focus distances from the metadata.
  - `imgSet.read_img(idx)`: Reads and post-processes a CR3 raw image.
  - `imgSet.get_gt_path()`: Returns the path to the calibration image used for this image set.
  - `imgSet.read_gt()`: Reads the corresponding calibration ground truth image.
  - `imgSet.get_pose()`: Returns the pose of the plane for this image set.
  - `imgSet.get_calib(idx)`: Returns the camera calibration for the depth of an image.
  - `imgSet.get_stack()`: Loads a pre-generated numpy array stack of grayscale images.
  - Defines `bokeh_img_sets` and `checkerboard_img_sets` dictionaries with image set configurations.

- **`mitsuba_simulation.py`**: Main script for rendering scenes using Mitsuba to simulate camera blur.
  - `load_first_pose(pose_file)`: Loads the first transformation matrix from pose estimations JSON.
  - `create_mitsuba_scene(matrix, calib_image_path, fov, focus_distance)`: Creates a Mitsuba scene dictionary with camera, rectangle, and lighting.
  - `main()`: Orchestrates loading pose, finding calibration image, computing FOV and depth, creating scene, and rendering.

- **`pose_estimation.py`**: Estimates camera poses using ArUco markers and PnP solving.
  - `generate_3d_from_json(detected_ids, json_path, meters_per_pixel)`: Generates 3D world coordinates from marker layout JSON.
  - `find_calib_photo_corners(image)`: Detects ArUco markers in calibration photos with sub-pixel refinement.
  - `solve_pnp(objpoints, imgpoints, K, D)`: Solves Perspective-n-Point problem to get camera transformation matrix.
  - Main script processes bokeh image sets, detects markers, solves poses, and saves to `pose_estimations.json`.

### Configuration and Setup Files

- **`requirements.txt`**: Lists Python dependencies (numpy, opencv, matplotlib, scipy, rawpy, exiv2, etc.).

- **`setup.py`**: 
  - Generates focus stacks for image sets by converting CR3 images to grayscale numpy arrays.
  - Calculates poses for all image sets

- **`pose_estimations.json`**: Output file containing estimated camera poses (transformation matrices) for each image set.

### Data Directories

- **`bokeh_calib_photos/`**: Contains subdirectories with CR3 raw images for bokeh calibration at different orientations and depths.

- **`calib_images/`**: Stores generated calibration images (PNG) and their marker layouts (JSON).

- **`checkerboard_images/`**: Contains subdirectories with CR3 images of checkerboard patterns at various depths.

- **`depth_groups/`**: Organized checkerboard images grouped by focus depth, with calibration JSON files.

## Setup

1. Create and enter virtual environment:
   ```
   py -m venv .venv
   .venv/Scripts/activate
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Mitsuba renderer (required for `mitsuba_simulation.py`):
   - Download and install Mitsuba 3 from https://www.mitsuba-renderer.org/
   - Ensure Mitsuba Python bindings are available in your environment
   - The script uses CUDA variant (`cuda_ad_rgb`), so CUDA-compatible GPU is recommended
   - note: installing requirements.txt should install mitsuba automatically

4. Place calibration data:
   - Put bokeh calibration photos in `/bokeh_calib_photos`
   - Put checkerboard images in `/checkerboard_images`

## Steps to Run mitsuba_simulation.py

1. **Group checkerboard images by depth**:
   ```
   python group_checkerboard_by_depth.py
   ```
   This creates `depth_groups/` with images organized by focus depth.

2. **Calibrate camera at each depth** (if not already done):
   Do not run if you don't have to.
   ```
   python calibrate_depths.py
   ```
   This generates `calib.json` files in each depth folder with camera intrinsics.

3. **Generate calibration images** (if not already done):
   ```
   python calib_images.py
   ```
   This creates PNG calibration images with ArUco markers in `calib_images/`.

4. **Estimate camera poses**:
   ```
   python pose_estimation.py
   ```
   This processes bokeh image sets, detects markers, solves for camera poses, and saves to `pose_estimations.json`.

5. **Generate focus stacks** (optional, for analysis):
   ```
   python setup.py
   ```
   This creates numpy stacks in `stacks/` for faster image processing.

6. **Run Mitsuba simulation**:
   ```
   python mitsuba_simulation.py
   ```
   This loads the first pose from `pose_estimations.json`, computes FOV and focus distance, creates a Mitsuba scene with the calibration board, and renders to `rendered_scene.png`.

**Note**: Ensure all data directories and files are present before running. The simulation requires CUDA for the `cuda_ad_rgb` variant. Adjust paths and parameters as needed for your specific camera and setup.