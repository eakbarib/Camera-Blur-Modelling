from calc_pose import calc_pose
from calibrate_depths import calibrate_depths
from imgSet import dot_stack_sets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup script')
    parser.add_argument('--take-poses', action='store_true', help='Run the pose calculations')
    parser.add_argument('--calibrate_cameras', default='store_true', help='Run the camera calibrations')
    parser.add_argument('--calibrate-skip', action='store_true', help='Skip already calculated calibrations')
    parser.add_argument('--calibrate-do-stacks', action='store_true', help='Calibrate stacks (only needed if doing analysis)')
    args = parser.parse_args()
    
    if args.take_poses:
        for img_set in dot_stack_sets.values():
            print(f"Taking pose of {img_set.id}")
            calc_pose(img_set)
    if args.calibrate_cameras:
        calibrate_depths(skip_complete=True, calibrate_stacks=False)