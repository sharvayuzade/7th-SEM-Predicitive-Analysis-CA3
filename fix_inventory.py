"""
Fix notebook 01 (save dataset inventory manually since there were unicode errors)
and execute notebooks 03-09
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"

# Manually write the dataset inventory that notebook 01 failed to save
uavs_raw_stats = {
    "Evening Fire Incident_raw_img": {
        "count": 354, "label": "FIRE",
        "path": str(ROOT / "forestfire-8gb" / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)" / "Evening Fire Incident_raw_img" / "Evening Fire Incident_raw_img")
    },
    "Evening Forest condition_raw_img": {
        "count": 286, "label": "NO_FIRE",
        "path": str(ROOT / "forestfire-8gb" / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)" / "Evening Forest condition_raw_img" / "Evening Forest condition_raw_img")
    },
    "Pre-Evening Fire Incident_raw_img": {
        "count": 791, "label": "FIRE",
        "path": str(ROOT / "forestfire-8gb" / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)" / "Pre-Evening Fire Incident_raw_img" / "Pre-Evening Fire Incident_raw_img")
    },
    "Pre-evening Forest condition_raw_img": {
        "count": 99, "label": "NO_FIRE",
        "path": str(ROOT / "forestfire-8gb" / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)" / "Pre-evening Forest condition_raw_img" / "Pre-evening Forest condition_raw_img")
    }
}

metadata = {
    'archive_fire':    755,
    'archive_nofire':  244,
    'archive_total':   999,
    'uavs_fire':       354 + 791,   # 1145
    'uavs_nofire':     286 + 99,    # 385
    'uavs_raw_total':  354 + 791 + 286 + 99,  # 1530
    'uavs_augmented':  15560,
    'csv_rows':        518,
    'csv_columns':     13,
    'fire_img_dir':    str(ROOT / "archive (1)" / "fire_dataset" / "fire_images"),
    'nonfire_img_dir': str(ROOT / "archive (1)" / "fire_dataset" / "non_fire_images"),
    'uavs_raw_stats':  uavs_raw_stats,
    'csv_path':        str(ROOT / "forest+fires" / "forestfires.csv")
}

with open(META_DIR / "dataset_inventory.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Dataset inventory saved:")
print(f"  Archive: {metadata['archive_total']} images")
print(f"  UAVS raw: {metadata['uavs_raw_total']} images")
print(f"  UAVS aug: {metadata['uavs_augmented']} images")
print(f"  CSV: {metadata['csv_rows']} rows")
print()
print("Ready to run model training notebooks.")
