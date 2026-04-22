from colmap import colmapDB
from imgSet import dot_stack_sets
from common import *

set_ids = ["or15_ir0_ds40", "or10_ir0_ds30", "or1_ir0_ds20"]

for base_path in [colmap_room_path, colmap_tabletop_path]:
    
    # reconstruction with vanilla colmap
    def_db = colmapDB(base_path / f"vanilla.db")
    def_db.add_images_auto(base_path / "images")
    def_db.register_images()
    def_db.match_images()
    def_db.reconstruct_sparse(base_path / f"vanilla_reconstruction", refine=True)
    reprojection_error = def_db.sparse_reproj_error()
    print(f"Reprojection errors on {base_path} with colmap determined parameters:")
    print(reprojection_error)
    
    def_db.reconstruct_dense(base_path / f"vanilla_dense", verbose=True)
    
    # reconstruction with fixed camera parameters
    fixed_db = colmapDB(base_path / f"fixed.db")

    def get_calibration_fixed(image_path):
        return load_calibration(checkerboard_single_path / "calib.json")

    fixed_db.add_images_cameras(base_path / "images", get_calibration_fixed)
    fixed_db.register_images()
    fixed_db.match_images()
    fixed_db.reconstruct_sparse(base_path / f"fixed_reconstruction")
    reprojection_error = fixed_db.sparse_reproj_error()
    print(f"Reprojection errors on {base_path} with fixed camera parameters:")
    print(reprojection_error)
    
    fixed_db.reconstruct_dense(base_path / f"fixed_dense")
    
    # reconstruction with optimized camera parameters
    for set_id in set_ids:
        img_set = dot_stack_sets[set_id]
        optim_db = colmapDB(base_path / f"{set_id}_optim.db")

        def get_calibration_optim(image_path):
            depth = read_image_depth_range(image_path)[2]
            return img_set.interpolate_calib(depth)

        optim_db.add_images_cameras(base_path / "images", get_calibration_optim)
        optim_db.register_images()
        optim_db.match_images()
        optim_db.reconstruct_sparse(base_path / f"{set_id}_optim_reconstruction")
        reprojection_error = optim_db.sparse_reproj_error()
        print(f"Reprojection errors for {set_id} on {base_path} with optimized camera parameters:")
        print(reprojection_error)
        
        optim_db.reconstruct_dense(base_path / f"{set_id}_optim_dense")