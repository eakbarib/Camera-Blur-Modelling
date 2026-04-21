from colmap import colmapDB
from imgSet import dot_stack_sets
from common import *

set_ids = ["or15_ir0_ds40", "or10_ir0_ds30", "or1_ir0_ds20"]
for set_id in set_ids:
    for base_path in [colmap_room_path, colmap_tabletop_path]:
        img_set = dot_stack_sets[set_id]
        db = colmapDB(base_path / f"{set_id}_optim.db")

        def get_calibration(image_path):
            depth = read_image_depth_range(image_path)[2]
            return img_set.interpolate_calib(depth)

        db.add_images_cameras(base_path / "images", get_calibration)
        db.register_images()
        db.match_images()
        db.reconstruct(base_path / f"{set_id}_optim_reconstruction")
        summary = db.summary()
        print(f"Reconstruction summary for {set_id} with optimized camera parameters:")
        print(summary)