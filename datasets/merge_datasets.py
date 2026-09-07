from pathlib import Path
import shutil
import yaml

# ============================================================
# PRAHARI DATASET MERGER
# Target classes:
# 0 = person
# 1 = handgun
# 2 = rifle
# 3 = knife
# 4 = grenade
# ============================================================

# Source datasets
DATASET1 = Path(r"C:\PRAHARI_DATA\dataset1\WEAPON_DETECTION_FINAL.v3i.yolov11")
DATASET2 = Path(r"C:\PRAHARI_DATA\dataset2\Yolo Weapon Detection.v1i.yolov11")

# Output dataset
OUTPUT = Path(r"C:\Users\mishm\PRAHARI\datasets\prahari")

# ------------------------------------------------------------
# Class mappings
# ------------------------------------------------------------

# Dataset 1:
# 0 Grenade -> 4
# 1 Gun     -> skip
# 2 Knife   -> 3
# 3 Pistol  -> 1

MAPPING_1 = {
    0: 4,
    1: None,
    2: 3,
    3: 1,
}

# Dataset 2:
# 0 Gunmen         -> skip
# 1 Rifle          -> 2
# 2 blunt object   -> skip
# 3 knife          -> 3
# 4 knife_attacker -> skip
# 5 person         -> 0
# 6 pistol         -> 1
# 7 shot-gun       -> skip
# 8 submachine-gun -> skip

MAPPING_2 = {
    0: None,
    1: 2,
    2: None,
    3: 3,
    4: None,
    5: 0,
    6: 1,
    7: None,
    8: None,
}

MAPPINGS = {
    "dataset1": MAPPING_1,
    "dataset2": MAPPING_2,
}

DATASETS = {
    "dataset1": DATASET1,
    "dataset2": DATASET2,
}

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

for split in ["train", "valid", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

stats = {
    "dataset1": {"train": 0, "valid": 0, "test": 0},
    "dataset2": {"train": 0, "valid": 0, "test": 0},
}

class_counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}

# ------------------------------------------------------------
# Process datasets
# ------------------------------------------------------------

for dataset_name, dataset_path in DATASETS.items():

    mapping = MAPPINGS[dataset_name]

    print("\n" + "=" * 60)
    print(f"Processing {dataset_name}")
    print(f"Location: {dataset_path}")
    print("=" * 60)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        continue

    for split in ["train", "valid", "test"]:

        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists():
            print(f"WARNING: Missing {image_dir}")
            continue

        if not label_dir.exists():
            print(f"WARNING: Missing {label_dir}")
            continue

        images = list(image_dir.glob("*"))

        for image_path in images:

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            ]:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            # Read annotations
            lines = label_path.read_text(encoding="utf-8").splitlines()

            converted_lines = []
            contains_unsupported = False

            for line in lines:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                old_class = int(parts[0])

                # Unsupported class found
                if old_class not in mapping or mapping[old_class] is None:
                    contains_unsupported = True
                    break

                new_class = mapping[old_class]

                converted_lines.append(
                    " ".join([str(new_class)] + parts[1:])
                )

                class_counts[new_class] += 1

            # Conservative policy:
            # If an image contains ANY unsupported object,
            # skip the entire image rather than creating
            # incomplete annotations.
            if contains_unsupported:
                # Roll back class counts from this image
                for converted_line in converted_lines:
                    cls = int(converted_line.split()[0])
                    class_counts[cls] -= 1

                continue

            if not converted_lines:
                continue

            # Prefix prevents filename collisions
            new_filename = f"{dataset_name}_{image_path.name}"

            output_image = OUTPUT / split / "images" / new_filename
            output_label = OUTPUT / split / "labels" / f"{Path(new_filename).stem}.txt"

            shutil.copy2(image_path, output_image)

            output_label.write_text(
                "\n".join(converted_lines) + "\n",
                encoding="utf-8"
            )

            stats[dataset_name][split] += 1


# ------------------------------------------------------------
# Create final data.yaml
# ------------------------------------------------------------

data_yaml = {
    "path": str(OUTPUT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 5,
    "names": [
        "person",
        "handgun",
        "rifle",
        "knife",
        "grenade"
    ]
}

with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

# ------------------------------------------------------------
# Print final report
# ------------------------------------------------------------

print("\n\n" + "=" * 60)
print("PRAHARI DATASET MERGE COMPLETE")
print("=" * 60)

print("\nImages successfully imported:")

for dataset_name in stats:
    print(f"\n{dataset_name}:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {stats[dataset_name][split]}")

print("\nFinal class annotation counts:")

class_names = {
    0: "person",
    1: "handgun",
    2: "rifle",
    3: "knife",
    4: "grenade",
}

for cls_id, count in class_counts.items():
    print(f"  {cls_id} = {class_names[cls_id]:10s}: {count}")

print("\nOutput:")
print(OUTPUT)

print("\nDataset structure:")
print("""
prahari/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
""")

print("Next step: inspect the statistics before training YOLO.")