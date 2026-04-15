from imgSet import checkerboard_img_sets

# tolerance for float comparision when grouping identical depth values
DEPTH_ROUND_DECIMALS = 6

# depth value -> list of (imgset_key, image_index, image_file_path, depth)
depth_groups = {}

for set_key, img_set in checkerboard_img_sets.items():
    for idx in range(img_set.count):
        fmin, fmax = img_set.get_focus_distance_range(idx)
        depth = float((fmin + fmax) / 2.0)
        depth_key = round(depth, DEPTH_ROUND_DECIMALS)
        image_name = f"IMG_{img_set.start + idx:04d}.JPG"
        image_path = f"{img_set.folder}/{img_set.id}/{image_name}"

        depth_groups.setdefault(depth_key, []).append({
            "imgset": set_key,
            "imgset_id": img_set.id,
            "index_in_set": idx,
            "image_name": image_name,
            "image_path": image_path,
            "depth": depth,
        })

# Build grouped distinct imgsets structure matching the same-depth grouping
grouped_imgsets = {
    depth: {
        "count": len(entries),
        "images": entries,
    }
    for depth, entries in sorted(depth_groups.items())
}

import os
import shutil

if __name__ == "__main__":
    print("Grouped checkerboard images by average depth (fmin/fmax midpoint):\n")
    for depth, group in grouped_imgsets.items():
        print(f"depth={depth:.6f}, count={group['count']}")
        for img in group["images"]:
            print(f"  - set {img['imgset']} ({img['imgset_id']}), idx={img['index_in_set']}, {img['image_name']}")

        print("\n")

    print(f"Total distinct depths: {len(grouped_imgsets)}")
    total_images = sum(g["count"] for g in grouped_imgsets.values())
    print(f"Total checkerboard images processed: {total_images}")

    # create depth-based directory structure and copy images
    output_root = os.path.join(os.getcwd(), "depth_groups")
    os.makedirs(output_root, exist_ok=True)

    print(f"\nCreating grouped folders under: {output_root}")
    for depth, group in sorted(grouped_imgsets.items(), key=lambda x: x[0]):
        depth_name = f"depth_{depth}"
        depth_dir = os.path.join(output_root, depth_name)
        os.makedirs(depth_dir, exist_ok=True)

        # rename each image to simple numeric names (1.JPG, 2.JPG, ...)
        count = 0
        for img in group["images"]:
            src_path = os.path.normpath(img["image_path"])
            if not os.path.isfile(src_path):
                print(f"Warning: source file not found: {src_path}")
                continue

            count += 1
            base_name = f"{count}.JPG"
            dst_path = os.path.join(depth_dir, base_name)

            # ensure deterministic overwrite behavior: keep first if exists
            if os.path.exists(dst_path):
                print(f"Skipping existing {dst_path}, keep first occurrence")
                continue

            shutil.move(src_path, dst_path)

        print(f"Saved {len([n for n in os.listdir(depth_dir) if n.endswith('.JPG')])} files to {depth_dir}")

    # delete any leftover duplicate-style names like 2_3.JPG, 2_4.JPG
    for depth_name in os.listdir(output_root):
        depth_dir = os.path.join(output_root, depth_name)
        if not os.path.isdir(depth_dir):
            continue
        for filename in os.listdir(depth_dir):
            if filename.endswith('.JPG') and '_' in filename:
                os.remove(os.path.join(depth_dir, filename))

    print("\nReorganization complete.")
