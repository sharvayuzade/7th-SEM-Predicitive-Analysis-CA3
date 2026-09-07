"""
Prepare YOLO Classification Dataset
====================================
Reads data_splits.json and creates the folder structure YOLO expects:
    data/yolo_cls/{train,val,test}/{FIRE,NO_FIRE}/

Uses hard-links (fast, no disk copy) with a fallback to file copy.
"""

import os
import json
import shutil
from pathlib import Path

IMPL = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task/Implementation")
META_DIR = IMPL / "artifacts" / "metadata"
OUT_DIR = IMPL / "data" / "yolo_cls"

# Load existing data splits
with open(META_DIR / "data_splits.json") as f:
    splits = json.load(f)

print("=" * 60)
print("PREPARING YOLO CLASSIFICATION DATASET")
print("=" * 60)

for split_name in ["train", "val", "test"]:
    items = splits[split_name]
    print(f"\n  {split_name}: {len(items)} images")

    for item in items:
        if isinstance(item, list):
            src_path, class_name = item[0], item[1]
        elif isinstance(item, dict):
            src_path = item.get("path", "")
            # Derive class from label index
            class_name = "FIRE" if item.get("label", 0) == 0 else "NO_FIRE"
        else:
            continue

        src = Path(src_path)
        if not src.exists():
            continue

        # Destination: data/yolo_cls/{split}/{CLASS}/{filename}
        # Use a unique name to avoid collisions across source directories
        unique_name = f"{hash(str(src)) & 0xFFFFFFFF:08x}_{src.name}"
        dst_dir = OUT_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / unique_name

        if dst.exists():
            continue

        # Try hard link first (instant, no extra disk), fall back to copy
        try:
            os.link(str(src), str(dst))
        except (OSError, NotImplementedError):
            shutil.copy2(str(src), str(dst))

# Print summary
print("\n" + "=" * 60)
print("DATASET STRUCTURE CREATED")
print("=" * 60)
for split_name in ["train", "val", "test"]:
    split_dir = OUT_DIR / split_name
    if split_dir.exists():
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*")))
                print(f"  {split_name}/{cls_dir.name}: {count} images")

print(f"\n  Output: {OUT_DIR}")
print("=" * 60)
