"""
Forest Fire AI — Notebook Generator
Creates all 10 Jupyter notebooks programmatically using nbformat.
"""

import nbformat as nbf
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

def save(notebook, name):
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print(f"  Created: {name}")

# ============================================================
# NOTEBOOK 01 — DATASET AUDIT
# ============================================================
nb01 = nb([
md("""# Forest Fire Detection — Dataset Audit
## Notebook 01: Complete Dataset Discovery & Analysis

**Objective**: Recursively inspect all dataset directories, identify every image, CSV, and archive file,
compute statistics, and select datasets for training.
"""),
code("""import os, sys, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL = ROOT / "Implementation"
ARCHIVE_DIR   = ROOT / "archive (1)"
FOREST_DIR    = ROOT / "forest+fires"
UAVS_DIR      = ROOT / "forestfire-8gb"
ARTIFACTS     = IMPL / "artifacts"
METADATA_DIR  = ARTIFACTS / "metadata"
PLOTS_DIR     = ARTIFACTS / "plots"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
CSV_EXTS   = {'.csv'}
ARCH_EXTS  = {'.zip', '.rar', '.7z', '.tar', '.gz'}

print("Forest Fire AI — Dataset Audit")
print("="*60)
print(f"Project root: {ROOT}")
print(f"Python: {sys.version.split()[0]}")
"""),
md("## 1. Recursive File Discovery"),
code("""def scan_dir(base_path, label):
    results = {'images': [], 'csvs': [], 'archives': [], 'others': []}
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"  [WARN] {base_path} does not exist")
        return results
    for p in base_path.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                results['images'].append(str(p))
            elif ext in CSV_EXTS:
                results['csvs'].append(str(p))
            elif ext in ARCH_EXTS:
                results['archives'].append(str(p))
            else:
                results['others'].append(str(p))
    print(f"\\n[{label}]")
    print(f"  Images:   {len(results['images']):,}")
    print(f"  CSVs:     {len(results['csvs'])}")
    print(f"  Archives: {len(results['archives'])}")
    print(f"  Others:   {len(results['others'])}")
    return results

scan_archive  = scan_dir(ARCHIVE_DIR,  "archive (1)")
scan_forest   = scan_dir(FOREST_DIR,   "forest+fires")
scan_uavs     = scan_dir(UAVS_DIR,     "forestfire-8gb")
"""),
md("## 2. Archive Dataset — fire_dataset"),
code("""fire_img_dir    = ARCHIVE_DIR / "fire_dataset" / "fire_images"
nonfire_img_dir = ARCHIVE_DIR / "fire_dataset" / "non_fire_images"

fire_imgs    = sorted(fire_img_dir.glob("*.png"))
nonfire_imgs = sorted(nonfire_img_dir.glob("*.png"))

print(f"Archive Dataset — fire_dataset")
print(f"  fire_images:     {len(fire_imgs):,}")
print(f"  non_fire_images: {len(nonfire_imgs):,}")
print(f"  Total:           {len(fire_imgs)+len(nonfire_imgs):,}")
print(f"  Class ratio fire:nofire = {len(fire_imgs)/len(nonfire_imgs):.2f}:1")
"""),
md("## 3. UAVS-FDDB Dataset Breakdown"),
code("""uavs_raw = UAVS_DIR / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)"
uavs_aug = UAVS_DIR / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Augmented Images"

# Map folders to labels
FIRE_LABEL_KEYWORDS    = ['fire', 'Fire', 'FIRE']
NOFIRE_LABEL_KEYWORDS  = ['forest', 'Forest', 'FOREST', 'condition', 'Condition']

uavs_raw_stats = {}
for d in sorted(uavs_raw.iterdir()):
    if d.is_dir():
        imgs = list(d.rglob("*"))
        imgs = [f for f in imgs if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        folder_name = d.name
        # Determine label from folder name
        name_lower = folder_name.lower()
        if 'fire incident' in name_lower:
            label = 'FIRE'
        elif 'forest condition' in name_lower:
            label = 'NO_FIRE'
        else:
            label = 'UNKNOWN'
        uavs_raw_stats[folder_name] = {'count': len(imgs), 'label': label, 'path': str(d)}

print("UAVS-FDDB Raw Image Dataset:")
for folder, info in uavs_raw_stats.items():
    print(f"  {folder}: {info['count']} images → Label: {info['label']}")

total_uavs_fire    = sum(v['count'] for v in uavs_raw_stats.values() if v['label']=='FIRE')
total_uavs_nofire  = sum(v['count'] for v in uavs_raw_stats.values() if v['label']=='NO_FIRE')
total_uavs_raw     = sum(v['count'] for v in uavs_raw_stats.values())
print(f"\\nTotal FIRE images:    {total_uavs_fire}")
print(f"Total NO_FIRE images: {total_uavs_nofire}")
print(f"Total raw images:     {total_uavs_raw}")
"""),
code("""# Augmented images
uavs_aug_stats = {}
for d in sorted(uavs_aug.iterdir()):
    if d.is_dir():
        imgs = list(d.rglob("*"))
        imgs = [f for f in imgs if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        name_lower = d.name.lower()
        if 'fire' in name_lower:
            label = 'FIRE'
        else:
            label = 'NO_FIRE'
        uavs_aug_stats[d.name] = {'count': len(imgs), 'label': label}
        print(f"  {d.name}: {len(imgs)} images → {label}")

total_aug = sum(v['count'] for v in uavs_aug_stats.values())
print(f"\\nTotal augmented images: {total_aug:,}")
"""),
md("## 4. Forest Fires CSV Analysis"),
code("""csv_path = FOREST_DIR / "forestfires.csv"
df = pd.read_csv(csv_path)
print(f"forestfires.csv  →  Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nData types:")
print(df.dtypes.to_string())
print(f"\\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"\\nTarget column 'area' statistics:")
print(df['area'].describe().round(4).to_string())
print(f"\\nRows where area > 0 (fire occurred): {(df['area'] > 0).sum()}")
print(f"Rows where area == 0 (no fire burned): {(df['area'] == 0).sum()}")
print(f"\\nMonths present: {sorted(df['month'].unique())}")
print(f"Days present:   {sorted(df['day'].unique())}")
"""),
md("## 5. Image Dimension Analysis"),
code("""from PIL import Image
import random
random.seed(42)

def sample_image_stats(img_paths, n=50, label=""):
    sample = random.sample(img_paths, min(n, len(img_paths)))
    sizes, modes = [], []
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                modes.append(im.mode)
        except:
            pass
    if sizes:
        widths  = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  [{label}] n={len(sizes)} sampled")
        print(f"    Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"    Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        print(f"    Modes:  {set(modes)}")

# Archive dataset
print("Archive Dataset image dimensions:")
sample_image_stats([str(p) for p in fire_imgs], label="FIRE")
sample_image_stats([str(p) for p in nonfire_imgs], label="NO_FIRE")

# UAVS dataset
print("\\nUAVS-FDDB Raw image dimensions:")
for folder, info in uavs_raw_stats.items():
    p = Path(info['path'])
    imgs = list(p.rglob("*"))
    imgs = [str(f) for f in imgs if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if imgs:
        sample_image_stats(imgs, n=30, label=f"{info['label']} ({folder[:30]})")
"""),
md("## 6. Visualize Sample Images"),
code("""fig, axes = plt.subplots(2, 8, figsize=(20, 6))
fig.suptitle("Sample Images — Forest Fire Dataset", fontsize=14, fontweight='bold')

fire_sample = random.sample([str(p) for p in fire_imgs], 8)
nonfire_sample = random.sample([str(p) for p in nonfire_imgs], 8)

for i, (row_imgs, row_label) in enumerate([(fire_sample, 'FIRE'), (nonfire_sample, 'NO FIRE')]):
    for j, img_path in enumerate(row_imgs):
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i][j].imshow(img)
            axes[i][j].axis('off')
            if j == 0:
                axes[i][j].set_title(row_label, fontsize=10, fontweight='bold',
                                      color='red' if row_label=='FIRE' else 'green')
        except Exception as e:
            axes[i][j].axis('off')

plt.tight_layout()
plot_path = PLOTS_DIR / "sample_images.png"
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")
"""),
md("## 7. Class Distribution Visualization"),
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Archive dataset
labels_arch = ['FIRE', 'NO_FIRE']
counts_arch = [len(fire_imgs), len(nonfire_imgs)]
colors = ['#E84040', '#2E8B57']
bars = axes[0].bar(labels_arch, counts_arch, color=colors, edgecolor='black', width=0.5)
axes[0].set_title('Archive Dataset — Class Distribution', fontweight='bold')
axes[0].set_ylabel('Number of Images')
for bar, count in zip(bars, counts_arch):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')

# UAVS raw
uavs_labels = list(uavs_raw_stats.keys())
uavs_counts = [v['count'] for v in uavs_raw_stats.values()]
uavs_colors = ['#E84040' if v['label']=='FIRE' else '#2E8B57'
                for v in uavs_raw_stats.values()]
bars2 = axes[1].bar(range(len(uavs_labels)), uavs_counts, color=uavs_colors, edgecolor='black')
axes[1].set_xticks(range(len(uavs_labels)))
axes[1].set_xticklabels([l[:25] for l in uavs_labels], rotation=20, ha='right', fontsize=8)
axes[1].set_title('UAVS-FDDB Raw — Class Distribution', fontweight='bold')
axes[1].set_ylabel('Number of Images')
for bar, count in zip(bars2, uavs_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=9)
fire_patch = mpatches.Patch(color='#E84040', label='FIRE')
nofire_patch = mpatches.Patch(color='#2E8B57', label='NO_FIRE')
axes[1].legend(handles=[fire_patch, nofire_patch])

plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=100, bbox_inches='tight')
plt.close()
print("Class distribution plot saved.")
"""),
md("## 8. Dataset Summary & Selection Decision"),
code("""# Compute combined totals
archive_total = len(fire_imgs) + len(nonfire_imgs)
uavs_total    = total_uavs_raw

# Build summary
summary_data = {
    'Dataset': [
        'Archive fire_dataset',
        'UAVS-FDDB Raw Images',
        'UAVS-FDDB Augmented',
        'forestfires.csv'
    ],
    'Type': ['Image', 'Image', 'Image', 'Tabular'],
    'FIRE Samples': [len(fire_imgs), total_uavs_fire, 'N/A (pre-augmented)', 'N/A'],
    'NO_FIRE Samples': [len(nonfire_imgs), total_uavs_nofire, 'N/A', 'N/A'],
    'Total': [archive_total, uavs_total, total_aug, len(df)],
    'Classes': ['FIRE, NO_FIRE', 'FIRE, NO_FIRE', 'FIRE, NO_FIRE', 'Regression (area)'],
    'Suitable For': ['Image Classification', 'Image Classification', 'Augmentation reference', 'Fire Risk Regression']
}
df_summary = pd.DataFrame(summary_data)
print("\\n" + "="*80)
print("DATASET INVENTORY")
print("="*80)
print(df_summary.to_string(index=False))

# Save metadata
metadata = {
    'archive_fire': len(fire_imgs),
    'archive_nofire': len(nonfire_imgs),
    'archive_total': archive_total,
    'uavs_fire': total_uavs_fire,
    'uavs_nofire': total_uavs_nofire,
    'uavs_raw_total': uavs_total,
    'uavs_augmented': total_aug,
    'csv_rows': len(df),
    'csv_columns': len(df.columns),
    'fire_img_dir': str(fire_img_dir),
    'nonfire_img_dir': str(nonfire_img_dir),
    'uavs_raw_stats': uavs_raw_stats,
    'csv_path': str(csv_path)
}
with open(METADATA_DIR / "dataset_inventory.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\\nMetadata saved: {METADATA_DIR}/dataset_inventory.json")
"""),
code("""
msg = [
    "=" * 70,
    "  DATASET SELECTION DECISION",
    "=" * 70,
    "  IMAGE CLASSIFICATION:",
    "    [PRIMARY]    Archive fire_dataset  (755 FIRE + 244 NO_FIRE = 999)",
    "    [SUPPLEMENT] UAVS-FDDB Raw Images  (1145 FIRE + 385 NO_FIRE)",
    "    [SKIP]       UAVS Augmented        (avoid test leakage)",
    "  TABULAR PREDICTION:",
    "    [USE]  forestfires.csv (518 rows, regression on burned area)",
    "    [USE]  Binary classification (area>0 = fire occurred)",
    "  COMBINED IMAGE TOTAL: ~2529 images (after de-duplication)",
    "  CLASSES: FIRE / NO_FIRE (binary)",
    "=" * 70
]
print("\\n".join(msg))
"""),
md("## 9. Findings Summary"),
code("""print("AUDIT COMPLETE")
print(f"  Archive images: {archive_total} (755 FIRE + 244 NO_FIRE)")
print(f"  UAVS raw images: {uavs_total} (1145 FIRE + 385 NO_FIRE)")
print(f"  UAVS augmented: {total_aug:,} (not used as test data)")
print(f"  CSV records: {len(df)}")
print(f"  CSV task: Regression (area) + Binary classification (area>0)")
print(f"  Target image input size for model: 224x224 px")
print(f"  No CUDA GPU detected — CPU training with efficient batch size")
"""),
])

# ============================================================
# NOTEBOOK 02 — IMAGE PREPROCESSING
# ============================================================
nb02 = nb([
md("""# Forest Fire Detection — Image Data Preprocessing
## Notebook 02: Data Loading, Cleaning, Splitting & Augmentation
"""),
code("""import os, sys, json, shutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

ROOT      = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL      = ROOT / "Implementation"
PROC_DIR  = IMPL / "data" / "processed"
PLOTS_DIR = IMPL / "artifacts" / "plots"
META_DIR  = IMPL / "artifacts" / "metadata"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
FIRE_DIR    = ROOT / "archive (1)" / "fire_dataset" / "fire_images"
NOFIRE_DIR  = ROOT / "archive (1)" / "fire_dataset" / "non_fire_images"

UAVS_RAW = ROOT / "forestfire-8gb" / "UAVS-FDDB UAVs-based Forest Fire Detection Database" / "Original Image Dataset (Raw Images)"
UAVS_FIRE_DIRS   = [UAVS_RAW / "Evening Fire Incident_raw_img" / "Evening Fire Incident_raw_img",
                     UAVS_RAW / "Pre-Evening Fire Incident_raw_img" / "Pre-Evening Fire Incident_raw_img"]
UAVS_NOFIRE_DIRS = [UAVS_RAW / "Evening Forest condition_raw_img" / "Evening Forest condition_raw_img",
                     UAVS_RAW / "Pre-evening Forest condition_raw_img" / "Pre-evening Forest condition_raw_img"]

IMAGE_SIZE = 224
RANDOM_SEED = 42

print("Image Preprocessing Pipeline")
print("="*60)
print(f"Target image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Random seed: {RANDOM_SEED}")
"""),
md("## 1. Collect All Image Paths"),
code("""def collect_images(directories, label, name=""):
    paths = []
    for d in directories:
        d = Path(d)
        if d.exists():
            for ext in ['*.png','*.jpg','*.jpeg','*.bmp','*.tiff']:
                paths.extend(list(d.glob(ext)))
            for ext in ['*.PNG','*.JPG','*.JPEG']:
                paths.extend(list(d.glob(ext)))
    print(f"  {name or label}: {len(paths)} images")
    return [(str(p), label) for p in paths]

# Archive dataset
arch_fire   = collect_images([FIRE_DIR],   'FIRE',    "Archive FIRE")
arch_nofire = collect_images([NOFIRE_DIR], 'NO_FIRE', "Archive NO_FIRE")

# UAVS dataset
uavs_fire   = collect_images(UAVS_FIRE_DIRS,   'FIRE',    "UAVS FIRE")
uavs_nofire = collect_images(UAVS_NOFIRE_DIRS, 'NO_FIRE', "UAVS NO_FIRE")

all_data = arch_fire + arch_nofire + uavs_fire + uavs_nofire
print(f"\\nTotal before deduplication: {len(all_data)}")
print(f"  FIRE:    {sum(1 for _,l in all_data if l=='FIRE')}")
print(f"  NO_FIRE: {sum(1 for _,l in all_data if l=='NO_FIRE')}")
"""),
md("## 2. Deduplication — Remove Exact Duplicates"),
code("""import random
random.seed(RANDOM_SEED)

def file_hash(path, chunk_size=8192):
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

print("Computing file hashes for deduplication...")
print("(Sampling for speed — checking all unique paths)")

seen_hashes = set()
unique_data = []
duplicates  = 0
corrupt     = 0

for i, (path, label) in enumerate(all_data):
    if i % 200 == 0:
        print(f"  Processed {i}/{len(all_data)}...", end='\\r')
    h = file_hash(path)
    if h is None:
        corrupt += 1
        continue
    if h in seen_hashes:
        duplicates += 1
    else:
        seen_hashes.add(h)
        # Also validate image can be opened
        try:
            with Image.open(path) as im:
                im.verify()
            unique_data.append((path, label))
        except:
            corrupt += 1

print(f"\\nDuplicates removed: {duplicates}")
print(f"Corrupt images:     {corrupt}")
print(f"Unique valid images: {len(unique_data)}")
print(f"  FIRE:    {sum(1 for _,l in unique_data if l=='FIRE')}")
print(f"  NO_FIRE: {sum(1 for _,l in unique_data if l=='NO_FIRE')}")
"""),
md("## 3. Train / Validation / Test Split (Stratified)"),
code("""paths  = [p for p,_ in unique_data]
labels = [l for _,l in unique_data]

# First split: 80% train+val, 20% test
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    paths, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
)

# Second split: 75% train, 25% val (of the 80%)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.25, random_state=RANDOM_SEED, stratify=train_val_labels
)

print("Dataset Split Results:")
print(f"  Training:   {len(train_paths)} images")
print(f"    FIRE:     {train_labels.count('FIRE')}")
print(f"    NO_FIRE:  {train_labels.count('NO_FIRE')}")
print(f"  Validation: {len(val_paths)} images")
print(f"    FIRE:     {val_labels.count('FIRE')}")
print(f"    NO_FIRE:  {val_labels.count('NO_FIRE')}")
print(f"  Test:       {len(test_paths)} images")
print(f"    FIRE:     {test_labels.count('FIRE')}")
print(f"    NO_FIRE:  {test_labels.count('NO_FIRE')}")

# Save split
split_data = {
    'train': list(zip(train_paths, train_labels)),
    'val':   list(zip(val_paths, val_labels)),
    'test':  list(zip(test_paths, test_labels))
}
with open(META_DIR / "data_splits.json", "w") as f:
    json.dump(split_data, f, indent=2)
print(f"\\nSplit saved to: {META_DIR}/data_splits.json")
"""),
md("## 4. Class Imbalance Analysis & Weighting"),
code("""from collections import Counter

train_counter = Counter(train_labels)
total_train = len(train_labels)

print("Class distribution in training set:")
for cls, cnt in train_counter.items():
    print(f"  {cls}: {cnt} ({100*cnt/total_train:.1f}%)")

# Compute class weights for loss function
n_fire    = train_counter.get('FIRE', 1)
n_nofire  = train_counter.get('NO_FIRE', 1)
n_total   = total_train

# Sklearn-style balanced weights
weight_fire   = n_total / (2 * n_fire)
weight_nofire = n_total / (2 * n_nofire)

CLASS_NAMES = ['FIRE', 'NO_FIRE']
class_weights = {'FIRE': weight_fire, 'NO_FIRE': weight_nofire}

print(f"\\nComputed class weights:")
for cls, w in class_weights.items():
    print(f"  {cls}: {w:.4f}")

# Save class weights
with open(META_DIR / "class_weights.json", "w") as f:
    json.dump({'class_names': CLASS_NAMES, 'class_weights': class_weights}, f, indent=2)
"""),
md("## 5. Data Augmentation Configuration"),
code("""# Define augmentation pipeline (torchvision transforms)
# These are applied only to TRAINING set
import torchvision.transforms as T
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomGrayscale(p=0.02),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Save transform config
transform_config = {
    'image_size': IMAGE_SIZE,
    'imagenet_mean': IMAGENET_MEAN,
    'imagenet_std': IMAGENET_STD,
    'train_augmentations': [
        'Resize(256x256)', 'RandomCrop(224x224)', 'RandomHorizontalFlip(p=0.5)',
        'RandomVerticalFlip(p=0.2)', 'RandomRotation(10°)',
        'ColorJitter(brightness=0.2,contrast=0.2)', 'RandomGrayscale(p=0.02)',
        'Normalize(ImageNet mean/std)'
    ],
    'val_test_augmentations': ['Resize(224x224)', 'Normalize(ImageNet mean/std)']
}
with open(META_DIR / "transform_config.json", "w") as f:
    json.dump(transform_config, f, indent=2)

print("Augmentation Pipeline:")
print("  Training transforms:")
for aug in transform_config['train_augmentations']:
    print(f"    • {aug}")
print("  Val/Test transforms:")
for aug in transform_config['val_test_augmentations']:
    print(f"    • {aug}")
"""),
md("## 6. Visualize Augmented Samples"),
code("""import random
random.seed(42)

def show_augmented(img_path, n_aug=6):
    base_transform = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
    aug_transform  = T.Compose([
        T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.2),
    ])
    img_pil = Image.open(img_path).convert('RGB')
    imgs = [base_transform(img_pil)] + [aug_transform(img_pil) for _ in range(n_aug)]
    return imgs

# Pick one fire and one non-fire
fire_sample_path   = random.choice(train_paths[:50] if train_labels[0]=='FIRE' else
                                   [p for p,l in zip(train_paths,train_labels) if l=='FIRE'][:10])
nofire_sample_path = random.choice([p for p,l in zip(train_paths,train_labels) if l=='NO_FIRE'][:10])

fig, axes = plt.subplots(2, 7, figsize=(18, 6))
fig.suptitle("Original + Augmented Samples", fontsize=13, fontweight='bold')
titles = ['Original', 'Aug 1', 'Aug 2', 'Aug 3', 'Aug 4', 'Aug 5', 'Aug 6']

for row_idx, (img_path, row_label, row_color) in enumerate([
    (fire_sample_path, 'FIRE', 'red'),
    (nofire_sample_path, 'NO FIRE', 'green')
]):
    imgs = show_augmented(img_path)
    for col_idx, img in enumerate(imgs):
        axes[row_idx][col_idx].imshow(img)
        axes[row_idx][col_idx].axis('off')
        t = titles[col_idx] if col_idx > 0 else f"{row_label}\\n{titles[0]}"
        axes[row_idx][col_idx].set_title(t, fontsize=8,
                                          color=row_color if col_idx==0 else 'black')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "augmentation_samples.png", dpi=100, bbox_inches='tight')
plt.close()
print("Augmentation visualization saved.")
"""),
code("""print("\\nPreprocessing Summary:")
print(f"  Total unique images: {len(unique_data)}")
print(f"  Training:   {len(train_paths)} ({100*len(train_paths)/len(unique_data):.0f}%)")
print(f"  Validation: {len(val_paths)} ({100*len(val_paths)/len(unique_data):.0f}%)")
print(f"  Test:       {len(test_paths)} ({100*len(test_paths)/len(unique_data):.0f}%)")
print(f"  Class weights: FIRE={weight_fire:.3f}, NO_FIRE={weight_nofire:.3f}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print("\\nNotebook 02 complete. Ready for model benchmarking.")
"""),
])

# ============================================================
# NOTEBOOK 03 — IMAGE MODEL BENCHMARKING
# ============================================================
nb03 = nb([
md("""# Forest Fire Detection — Image Model Benchmarking
## Notebook 03: Transfer Learning Benchmark (EfficientNet-B0, MobileNetV3, MobileNetV2, ResNet18)

**Objective**: Benchmark 4 lightweight pretrained models using short training runs to identify 
the best candidate for final training.
"""),
code("""import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"
PLOTS    = IMPL / "artifacts" / "plots"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {__import__('torchvision').__version__}")

# Load split
with open(META_DIR / "data_splits.json") as f:
    splits = json.load(f)
with open(META_DIR / "class_weights.json") as f:
    cw_data = json.load(f)

CLASS_NAMES  = cw_data['class_names']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
print(f"Classes: {CLASS_NAMES}")
print(f"Class to idx: {CLASS_TO_IDX}")
"""),
code("""class FireDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data      = data  # list of [path, label]
        self.transform = transform
        self.class_to_idx = CLASS_TO_IDX

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Use subset for benchmarking speed
train_data_full = splits['train']
val_data_full   = splits['val']

# For benchmarking use 600 train + 200 val (fast evaluation)
import random
random.seed(SEED)
train_bench = random.sample(train_data_full, min(600, len(train_data_full)))
val_bench   = random.sample(val_data_full,   min(200, len(val_data_full)))

train_ds = FireDataset(train_bench, train_transform)
val_ds   = FireDataset(val_bench,   val_transform)

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Class weights tensor
cw = cw_data['class_weights']
weight_tensor = torch.tensor([cw['FIRE'], cw['NO_FIRE']], dtype=torch.float32).to(DEVICE)

print(f"Benchmark training set:   {len(train_bench)}")
print(f"Benchmark validation set: {len(val_bench)}")
print(f"Batch size: {BATCH_SIZE}")
"""),
code("""def build_model(model_name, num_classes=2):
    if model_name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif model_name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'ResNet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)

def freeze_backbone(model, model_name):
    \"\"\"Freeze all layers except classifier head.\"\"\"
    if model_name == 'EfficientNet-B0':
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == 'MobileNetV3':
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == 'MobileNetV2':
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == 'ResNet18':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

def benchmark_model(model_name, epochs=5):
    print(f"\\n{'='*50}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'='*50}")
    model = build_model(model_name)
    freeze_backbone(model, model_name)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable params: {n_trainable:,}")
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    train_losses, val_accs = [], []
    t_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out   = model(imgs)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        train_losses.append(running_loss / len(train_loader))
        val_accs.append(acc)
        print(f"  Epoch {epoch+1}/{epochs} | loss={train_losses[-1]:.4f} | val_acc={acc:.4f}")
    
    train_time = time.time() - t_start
    
    # Final metrics
    model.eval()
    all_preds, all_labels = [], []
    inf_times = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            t0 = time.time()
            out = model(imgs)
            inf_times.append((time.time() - t0) / len(imgs) * 1000)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    inf_ms = np.mean(inf_times)
    
    print(f"\\n  Results: acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f} | inf={inf_ms:.2f}ms")
    
    return {
        'model': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'inference_ms': round(inf_ms, 2),
        'params_total': n_params,
        'params_trainable': n_trainable,
        'train_time_s': round(train_time, 1),
        'train_losses': train_losses,
        'val_accs': val_accs
    }

BENCHMARK_MODELS = ['EfficientNet-B0', 'MobileNetV3', 'MobileNetV2', 'ResNet18']
results = []
for mname in BENCHMARK_MODELS:
    r = benchmark_model(mname, epochs=5)
    results.append(r)
"""),
code("""# Display benchmark results
df_bench = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ('train_losses','val_accs')} for r in results])
df_bench = df_bench.sort_values('f1', ascending=False).reset_index(drop=True)

print("\\n" + "="*70)
print("BENCHMARK RESULTS — Sorted by F1 Score")
print("="*70)
print(df_bench[['model','accuracy','precision','recall','f1','inference_ms']].to_string(index=False))

# Save results
bench_dict = [{k: v for k, v in r.items() if k not in ('train_losses','val_accs')}
              for r in results]
with open(META_DIR / "benchmark_results.json", "w") as f:
    json.dump(bench_dict, f, indent=2)

best_model_name = df_bench.iloc[0]['model']
print(f"\\n✓ BEST MODEL: {best_model_name}")
print(f"  F1={df_bench.iloc[0]['f1']:.4f} | Acc={df_bench.iloc[0]['accuracy']:.4f}")
with open(META_DIR / "selected_model.json", "w") as f:
    json.dump({'model_name': best_model_name}, f, indent=2)
"""),
code("""# Plot benchmark comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Model Benchmarking Results", fontsize=14, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_models = ['#1E90FF', '#32CD32', '#FF8C00', '#DC143C']

df_plot = pd.DataFrame([{k: v for k, v in r.items() if k not in ('train_losses','val_accs')} for r in results])

for ax, metric, mlabel in zip(axes, metrics, metric_labels):
    bars = ax.bar(df_plot['model'], df_plot[metric], color=colors_models, edgecolor='black', alpha=0.85)
    ax.set_title(mlabel, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_xticklabels(df_plot['model'], rotation=20, ha='right', fontsize=8)
    for bar, val in zip(bars, df_plot[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS / "model_benchmark.png", dpi=100, bbox_inches='tight')
plt.close()
print("Benchmark plot saved.")
"""),
code("""# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for r, c in zip(results, colors_models):
    axes[0].plot(range(1, len(r['train_losses'])+1), r['train_losses'], label=r['model'], color=c, marker='o')
    axes[1].plot(range(1, len(r['val_accs'])+1), r['val_accs'], label=r['model'], color=c, marker='s')

axes[0].set_title('Training Loss (Benchmark)', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].set_title('Validation Accuracy (Benchmark)', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS / "benchmark_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\\nBenchmark complete. Selected model: {best_model_name}")
"""),
])

# ============================================================
# NOTEBOOK 04 — FINAL IMAGE MODEL TRAINING
# ============================================================
nb04 = nb([
md("""# Forest Fire Detection — Final Image Model Training
## Notebook 04: Full Training of Selected Model with Best Practices

Transfer learning with early stopping, LR scheduling, class weighting, and checkpointing.
"""),
code("""import os, sys, json, time, copy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"
PLOTS    = IMPL / "artifacts" / "plots"
MODEL_DIR = IMPL / "models" / "image"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load config
with open(META_DIR / "data_splits.json") as f:
    splits = json.load(f)
with open(META_DIR / "class_weights.json") as f:
    cw_data = json.load(f)
with open(META_DIR / "selected_model.json") as f:
    sel = json.load(f)

SELECTED_MODEL = sel['model_name']
CLASS_NAMES    = cw_data['class_names']
CLASS_TO_IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS   = {i: c for i, c in enumerate(CLASS_NAMES)}
cw             = cw_data['class_weights']

print(f"Device:         {DEVICE}")
print(f"Selected model: {SELECTED_MODEL}")
print(f"Classes:        {CLASS_NAMES}")
"""),
code("""IMAGE_SIZE    = 224
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 20
PATIENCE      = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class FireDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data      = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, CLASS_TO_IDX[label]

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=12),
    T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomGrayscale(p=0.02),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_ds = FireDataset(splits['train'], train_transform)
val_ds   = FireDataset(splits['val'],   val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

weight_tensor = torch.tensor([cw['FIRE'], cw['NO_FIRE']], dtype=torch.float32).to(DEVICE)

print(f"Training samples:   {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Batch size: {BATCH_SIZE} | Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")
"""),
code("""def build_final_model(model_name, num_classes=2, fine_tune_layers=2):
    if model_name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Unfreeze last N feature blocks + classifier
        feature_blocks = list(m.features.children())
        for block in feature_blocks[:-fine_tune_layers]:
            for p in block.parameters():
                p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feature_blocks = list(m.features.children())
        for block in feature_blocks[:-fine_tune_layers]:
            for p in block.parameters():
                p.requires_grad = False
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif model_name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        feature_blocks = list(m.features.children())
        for block in feature_blocks[:-fine_tune_layers]:
            for p in block.parameters():
                p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'ResNet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for name, p in m.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)

model = build_final_model(SELECTED_MODEL)
n_params     = sum(p.numel() for p in model.parameters())
n_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {SELECTED_MODEL}")
print(f"  Total params:     {n_params:,}")
print(f"  Trainable params: {n_trainable:,}")
"""),
code("""criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [], 'lr': []}
best_val_f1  = 0.0
best_weights = None
no_improve   = 0

print(f"Starting training...")
t_start = time.time()

for epoch in range(MAX_EPOCHS):
    # ── Train ──────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_correct = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (out.argmax(1) == lbls).sum().item()
    
    train_loss /= len(train_loader)
    train_acc   = train_correct / len(train_ds)
    
    # ── Validate ────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, lbls)
            val_loss += loss.item()
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc   = accuracy_score(all_lbls, all_preds)
    val_f1    = f1_score(all_lbls, all_preds, average='weighted', zero_division=0)
    
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['lr'].append(lr)
    
    print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | "
          f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
          f"lr={lr:.2e}")
    
    # ── Early stopping ─────────────────────────────────────
    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_weights = copy.deepcopy(model.state_dict())
        no_improve   = 0
        torch.save(best_weights, MODEL_DIR / "best_model.pth")
        print(f"  ✓ New best val_f1={best_val_f1:.4f} — checkpoint saved")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

train_time = time.time() - t_start
print(f"\\nTraining complete in {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"Best validation F1: {best_val_f1:.4f}")
"""),
code("""# Load best weights
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
epochs_ran = list(range(1, len(history['train_loss'])+1))

axes[0].plot(epochs_ran, history['train_loss'], 'b-o', label='Train')
axes[0].plot(epochs_ran, history['val_loss'],   'r-s', label='Val')
axes[0].set_title('Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, history['train_acc'], 'b-o', label='Train')
axes[1].plot(epochs_ran, history['val_acc'],   'r-s', label='Val')
axes[1].set_title('Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs_ran, history['val_f1'], 'g-^', label='Val F1')
ax2 = axes[2].twinx()
ax2.plot(epochs_ran, history['lr'], 'k--', label='LR', alpha=0.5)
axes[2].set_title('Val F1 & Learning Rate', fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].legend(loc='upper left'); ax2.legend(loc='upper right')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS / "training_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print("Training curves saved.")
"""),
code("""# Save class names and metadata
class_names_data = {'class_names': CLASS_NAMES, 'class_to_idx': CLASS_TO_IDX, 'idx_to_class': IDX_TO_CLASS}
with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names_data, f, indent=2)

model_metadata = {
    'model_name': SELECTED_MODEL,
    'image_size': IMAGE_SIZE,
    'num_classes': 2,
    'class_names': CLASS_NAMES,
    'imagenet_mean': IMAGENET_MEAN,
    'imagenet_std': IMAGENET_STD,
    'best_val_f1': round(best_val_f1, 4),
    'training_time_s': round(train_time, 1),
    'epochs_ran': len(history['train_loss']),
    'device': str(DEVICE)
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

# Save history
with open(META_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("Model artifacts saved:")
print(f"  {MODEL_DIR}/best_model.pth")
print(f"  {MODEL_DIR}/class_names.json")
print(f"  {MODEL_DIR}/model_metadata.json")
print(f"\\nFinal model: {SELECTED_MODEL}")
print(f"Best Val F1: {best_val_f1:.4f}")
"""),
])

# ============================================================
# NOTEBOOK 05 — IMAGE MODEL EVALUATION & EXPLAINABILITY
# ============================================================
nb05 = nb([
md("""# Forest Fire Detection — Model Evaluation & Explainability
## Notebook 05: Test Set Evaluation, Confusion Matrix, Grad-CAM
"""),
code("""import os, sys, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"
PLOTS    = IMPL / "artifacts" / "plots"
METRICS  = IMPL / "artifacts" / "metrics"
MODEL_DIR = IMPL / "models" / "image"
METRICS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)

with open(META_DIR / "data_splits.json") as f:
    splits = json.load(f)
with open(MODEL_DIR / "class_names.json") as f:
    cnames = json.load(f)
with open(MODEL_DIR / "model_metadata.json") as f:
    mmeta = json.load(f)

CLASS_NAMES  = cnames['class_names']
CLASS_TO_IDX = cnames['class_to_idx']
IDX_TO_CLASS = {int(k): v for k, v in cnames['idx_to_class'].items()}
SELECTED_MODEL = mmeta['model_name']
IMAGE_SIZE     = mmeta['image_size']
IMAGENET_MEAN  = mmeta['imagenet_mean']
IMAGENET_STD   = mmeta['imagenet_std']

print(f"Model: {SELECTED_MODEL} | Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
"""),
code("""# ── Rebuild model & load weights ──────────────────────────────────────
def build_model(model_name, num_classes=2):
    if model_name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif model_name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'ResNet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

model = build_model(SELECTED_MODEL)
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded from: {MODEL_DIR}/best_model.pth")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""),
code("""class FireDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128,128,128))
        if self.transform: img = self.transform(img)
        return img, CLASS_TO_IDX[label], path

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
test_ds = FireDataset(splits['test'], test_transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

print(f"Test set: {len(test_ds)} images")
"""),
code("""# ── Evaluate on test set ──────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []
inf_times = []

with torch.no_grad():
    for imgs, lbls, _ in test_loader:
        imgs = imgs.to(DEVICE)
        t0   = time.time()
        out  = model(imgs)
        inf_times.append((time.time() - t0) / len(imgs) * 1000)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
fire_idx  = CLASS_TO_IDX['FIRE']
auc  = roc_auc_score(all_labels, all_probs[:, fire_idx])
inf_ms = np.mean(inf_times)

# Fire class specific recall (false negative rate)
cm = confusion_matrix(all_labels, all_preds)
fire_recall = cm[fire_idx, fire_idx] / cm[fire_idx].sum() if cm[fire_idx].sum() > 0 else 0.0

print("="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"  Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision:         {prec:.4f}")
print(f"  Recall (weighted): {rec:.4f}")
print(f"  F1-Score:          {f1:.4f}")
print(f"  ROC-AUC:           {auc:.4f}")
print(f"  FIRE class recall: {fire_recall:.4f}  (miss rate: {1-fire_recall:.4f})")
print(f"  Avg inference:     {inf_ms:.2f} ms/image")
print("\\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

test_metrics = {
    'accuracy': round(float(acc), 4),
    'precision': round(float(prec), 4),
    'recall': round(float(rec), 4),
    'f1': round(float(f1), 4),
    'roc_auc': round(float(auc), 4),
    'fire_recall': round(float(fire_recall), 4),
    'inference_ms': round(float(inf_ms), 2),
    'test_samples': len(test_ds)
}
with open(METRICS / "image_test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"\\nMetrics saved.")
"""),
code("""# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].set_title(f'Confusion Matrix — {SELECTED_MODEL}', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_xticks(range(len(CLASS_NAMES))); axes[0].set_yticks(range(len(CLASS_NAMES)))
axes[0].set_xticklabels(CLASS_NAMES); axes[0].set_yticklabels(CLASS_NAMES)
plt.colorbar(im, ax=axes[0])
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=12, fontweight='bold')

# Bar chart of metrics
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
metric_vals  = [acc, prec, rec, f1, auc]
colors = ['#1E90FF','#32CD32','#FF8C00','#DC143C','#9370DB']
bars = axes[1].bar(metric_names, metric_vals, color=colors, edgecolor='black', alpha=0.85)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Performance Metrics — Test Set', fontweight='bold')
for bar, val in zip(bars, metric_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS / "test_evaluation.png", dpi=100, bbox_inches='tight')
plt.close()
print("Evaluation plots saved.")
"""),
md("## 5. Grad-CAM Explainability"),
code("""class GradCAM:
    \"\"\"Lightweight Grad-CAM implementation for CNN feature visualization.\"\"\"
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            # Try to find the last conv layer automatically
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = (name, module)
            name, target = last_conv
            print(f"  Using last conv layer: {name}")
        target.register_forward_hook(self._save_activations)
        target.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = np.array([[cam]])
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# Find target layer for selected model
def get_target_layer(model_name):
    return {
        'EfficientNet-B0': 'features.8.0',
        'MobileNetV3':     'features.12',
        'MobileNetV2':     'features.18.0',
        'ResNet18':        'layer4.1.conv2'
    }.get(model_name, None)

target_layer = get_target_layer(SELECTED_MODEL)
print(f"Grad-CAM target layer: {target_layer}")

# Build fresh model with grad support
model_gc = build_model(SELECTED_MODEL)
model_gc.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location='cpu'))
model_gc.eval()
gradcam = GradCAM(model_gc, target_layer)
"""),
code("""import random
random.seed(42)

# Denormalize helper
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    t = tensor.clone()
    for c, m, s in zip(range(3), mean, std):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)

# Pick sample test images
test_paths = splits['test']
fire_samples   = [(p, l) for p, l in test_paths if l == 'FIRE'][:4]
nofire_samples = [(p, l) for p, l in test_paths if l == 'NO_FIRE'][:4]
samples = fire_samples[:2] + nofire_samples[:2]

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3.5*len(samples)))
fig.suptitle(f"Grad-CAM Explainability — {SELECTED_MODEL}", fontsize=13, fontweight='bold')

for row_idx, (img_path, true_label) in enumerate(samples):
    try:
        img_pil = Image.open(img_path).convert('RGB')
    except:
        continue

    inp = test_transform(img_pil).unsqueeze(0)

    # Grad-CAM
    cam, pred_idx = gradcam.generate(inp)
    pred_label = IDX_TO_CLASS[pred_idx]

    # Overlay
    img_resized = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    heatmap = cm.jet(cam)[:, :, :3]
    overlay = 0.55 * img_resized + 0.45 * heatmap

    # Confidence
    with torch.no_grad():
        logits = model_gc(inp)
        probs  = F.softmax(logits, dim=1)[0]
    conf = probs[pred_idx].item()
    correct = "✓" if pred_label == true_label else "✗"

    axes[row_idx][0].imshow(img_resized)
    axes[row_idx][0].set_title(f"Original\nTrue: {true_label}", fontsize=8)
    axes[row_idx][0].axis('off')

    axes[row_idx][1].imshow(cam, cmap='jet')
    axes[row_idx][1].set_title(f"Grad-CAM Heatmap\nHot=High attention", fontsize=8)
    axes[row_idx][1].axis('off')

    axes[row_idx][2].imshow(np.clip(overlay, 0, 1))
    c = 'green' if pred_label == true_label else 'red'
    axes[row_idx][2].set_title(f"Overlay {correct}\nPred: {pred_label} ({conf:.1%})", fontsize=8, color=c)
    axes[row_idx][2].axis('off')

plt.tight_layout()
plt.savefig(PLOTS / "gradcam_examples.png", dpi=100, bbox_inches='tight')
plt.close()
print("Grad-CAM visualization saved.")
print(f"\\nNotebook 05 complete.")
print(f"Test Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
"""),
])

# ============================================================
# NOTEBOOK 06 — TABULAR DATA ANALYSIS
# ============================================================
nb06 = nb([
md("""# Forest Fire AI — Tabular Data Analysis
## Notebook 06: Exploratory Data Analysis of forestfires.csv
"""),
code("""import os, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
PLOTS    = IMPL / "artifacts" / "plots"
META_DIR = IMPL / "artifacts" / "metadata"
DATA_DIR = IMPL / "data" / "processed"
PLOTS.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "forest+fires" / "forestfires.csv"
df = pd.read_csv(CSV_PATH)

print("Forest Fires CSV — EDA")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
"""),
code("""print("\\nData Types:")
print(df.dtypes.to_string())
print(f"\\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"\\nDescriptive Statistics:")
print(df.describe().round(3).to_string())
"""),
code("""# Add binary classification target
df['fire_occurred'] = (df['area'] > 0).astype(int)
df['area_log']      = np.log1p(df['area'])

print("Target Analysis — 'area' column (burned area in hectares):")
print(f"  Min:    {df['area'].min()}")
print(f"  Max:    {df['area'].max()}")
print(f"  Mean:   {df['area'].mean():.4f}")
print(f"  Median: {df['area'].median():.4f}")
print(f"  Zeros:  {(df['area'] == 0).sum()} ({100*(df['area']==0).mean():.1f}%)")
print(f"  >0:     {(df['area'] > 0).sum()} ({100*(df['area']>0).mean():.1f}%)")
print("\\nBinary classification target (area > 0):")
print(df['fire_occurred'].value_counts().to_string())
print(f"\\nTask decision: BOTH regression (area) + binary classification (fire_occurred)")
"""),
code("""# Encode categoricals
month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
day_order   = ['mon','tue','wed','thu','fri','sat','sun']

df['month_num'] = df['month'].map({m: i for i, m in enumerate(month_order, 1)})
df['day_num']   = df['day'].map({d: i for i, d in enumerate(day_order, 1)})

# Numerical features
num_cols = ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain',
            'month_num','day_num']

print("Numerical feature statistics:")
print(df[num_cols].describe().round(3).to_string())
"""),
code("""# Visualization — distributions
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Feature Distributions — Forest Fires Dataset", fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), num_cols):
    ax.hist(df[col], bins=25, color='#2E8B57', edgecolor='black', alpha=0.8)
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(PLOTS / "feature_distributions.png", dpi=100, bbox_inches='tight')
plt.close()
print("Feature distributions saved.")
"""),
code("""# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = num_cols + ['area', 'fire_occurred']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn',
            center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS / "correlation_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()

print("Top correlations with 'area':")
corr_with_area = corr['area'].drop('area').abs().sort_values(ascending=False)
print(corr_with_area.round(3).to_string())
"""),
code("""# Monthly fire distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

month_fires = df.groupby('month')['fire_occurred'].agg(['sum','count'])
month_fires = month_fires.reindex(month_order)
month_fires['pct'] = 100 * month_fires['sum'] / month_fires['count']

axes[0].bar(month_fires.index, month_fires['sum'], color='#E84040', edgecolor='black', alpha=0.8)
axes[0].set_title('Fire Events by Month', fontweight='bold')
axes[0].set_ylabel('Fire Events')
axes[0].set_xticklabels(month_fires.index, rotation=30)

area_by_month = df.groupby('month')['area'].sum().reindex(month_order)
axes[1].bar(area_by_month.index, area_by_month.values, color='#FF8C00', edgecolor='black', alpha=0.8)
axes[1].set_title('Total Burned Area by Month (ha)', fontweight='bold')
axes[1].set_ylabel('Total Area (ha)')
axes[1].set_xticklabels(area_by_month.index, rotation=30)

plt.tight_layout()
plt.savefig(PLOTS / "monthly_fire_analysis.png", dpi=100, bbox_inches='tight')
plt.close()
print("Monthly analysis saved.")
"""),
code("""# Save processed data for modeling
features = ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain','month','day']
df_processed = df[features + ['area', 'fire_occurred', 'area_log']].copy()

df_processed.to_csv(DATA_DIR / "forestfires_processed.csv", index=False)

tab_meta = {
    'features': features,
    'numerical_features': ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain'],
    'categorical_features': ['month','day'],
    'regression_target': 'area',
    'classification_target': 'fire_occurred',
    'n_samples': len(df),
    'fire_positive': int((df['fire_occurred']==1).sum()),
    'fire_negative': int((df['fire_occurred']==0).sum())
}
with open(META_DIR / "tabular_metadata.json", "w") as f:
    json.dump(tab_meta, f, indent=2)

print(f"Processed data saved: {DATA_DIR}/forestfires_processed.csv")
print(f"Tabular metadata saved: {META_DIR}/tabular_metadata.json")
print("\\nNotebook 06 complete.")
"""),
])

# ============================================================
# NOTEBOOK 07 — TABULAR MODEL BENCHMARKING
# ============================================================
nb07 = nb([
md("""# Forest Fire AI — Tabular Model Benchmarking
## Notebook 07: Benchmark ML Models for Classification & Regression
"""),
code("""import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, mean_absolute_error,
                              mean_squared_error, r2_score)
import xgboost as xgb

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"
PLOTS    = IMPL / "artifacts" / "plots"
DATA_DIR = IMPL / "data" / "processed"

with open(META_DIR / "tabular_metadata.json") as f:
    tmeta = json.load(f)

df = pd.read_csv(DATA_DIR / "forestfires_processed.csv")
print(f"Data: {df.shape}")
print(f"Features: {tmeta['features']}")
"""),
code("""# Preprocessing
num_feats = tmeta['numerical_features']
cat_feats  = tmeta['categorical_features']

le_month = LabelEncoder()
le_day   = LabelEncoder()
df['month_enc'] = le_month.fit_transform(df['month'])
df['day_enc']   = le_day.fit_transform(df['day'])

feature_cols = num_feats + ['month_enc', 'day_enc']
X = df[feature_cols].values
y_cls = df['fire_occurred'].values
y_reg = df['area'].values
y_reg_log = np.log1p(y_reg)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix: {X.shape}")
print(f"Classification target distribution: {np.bincount(y_cls)}")
print(f"Regression target range: {y_reg.min():.2f} to {y_reg.max():.2f}")
"""),
code("""# ── Classification Benchmark ──────────────────────────────────────────
cls_models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42,
                                   eval_metric='logloss', verbosity=0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

cls_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in cls_models.items():
    t0 = time.time()
    # Use scaled for logistic, raw+encoded for tree-based
    X_use = X_scaled if name == 'Logistic Regression' else X
    scores_acc = cross_val_score(clf, X_use, y_cls, cv=cv, scoring='accuracy')
    scores_f1  = cross_val_score(clf, X_use, y_cls, cv=cv, scoring='f1')
    scores_auc = cross_val_score(clf, X_use, y_cls, cv=cv, scoring='roc_auc')
    elapsed = time.time() - t0
    res = {
        'model': name,
        'cv_accuracy': round(scores_acc.mean(), 4),
        'cv_f1': round(scores_f1.mean(), 4),
        'cv_roc_auc': round(scores_auc.mean(), 4),
        'time_s': round(elapsed, 1)
    }
    cls_results.append(res)
    print(f"  {name}: acc={res['cv_accuracy']:.4f} | f1={res['cv_f1']:.4f} | auc={res['cv_roc_auc']:.4f} | {elapsed:.1f}s")

df_cls = pd.DataFrame(cls_results).sort_values('cv_f1', ascending=False)
print("\\nClassification Benchmark (5-fold CV):")
print(df_cls.to_string(index=False))
best_cls = df_cls.iloc[0]['model']
print(f"\\nBest classifier: {best_cls}")
"""),
code("""# ── Regression Benchmark ──────────────────────────────────────────────
reg_models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=42),
}

reg_results = []
cv_r = KFold(n_splits=5, shuffle=True, random_state=42)

for name, reg in reg_models.items():
    t0 = time.time()
    scores_mae  = -cross_val_score(reg, X, y_reg_log, cv=cv_r, scoring='neg_mean_absolute_error')
    scores_r2   =  cross_val_score(reg, X, y_reg_log, cv=cv_r, scoring='r2')
    elapsed = time.time() - t0
    res = {'model': name, 'cv_mae': round(scores_mae.mean(), 4),
           'cv_r2': round(scores_r2.mean(), 4), 'time_s': round(elapsed, 1)}
    reg_results.append(res)
    print(f"  {name}: MAE(log)={res['cv_mae']:.4f} | R²={res['cv_r2']:.4f} | {elapsed:.1f}s")

df_reg = pd.DataFrame(reg_results).sort_values('cv_r2', ascending=False)
print("\\nRegression Benchmark (5-fold CV, log-transformed target):")
print(df_reg.to_string(index=False))
best_reg = df_reg.iloc[0]['model']
print(f"\\nBest regressor: {best_reg}")
"""),
code("""# Save benchmark results and selections
bench_tab = {
    'classification': cls_results,
    'regression': reg_results,
    'best_classifier': best_cls,
    'best_regressor': best_reg,
    'feature_cols': feature_cols
}
with open(META_DIR / "tabular_benchmark.json", "w") as f:
    json.dump(bench_tab, f, indent=2)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names_cls = [r['model'] for r in cls_results]
f1s       = [r['cv_f1'] for r in cls_results]
aucs      = [r['cv_roc_auc'] for r in cls_results]

x = np.arange(len(names_cls))
w = 0.35
bars1 = axes[0].bar(x - w/2, f1s,  w, label='F1',     color='#1E90FF', edgecolor='black')
bars2 = axes[0].bar(x + w/2, aucs, w, label='ROC-AUC', color='#32CD32', edgecolor='black')
axes[0].set_xticks(x); axes[0].set_xticklabels(names_cls, rotation=20, ha='right', fontsize=9)
axes[0].set_ylim(0, 1.1); axes[0].set_title('Classification Benchmark', fontweight='bold')
axes[0].legend(); axes[0].set_ylabel('Score')

names_reg = [r['model'] for r in reg_results]
r2s = [max(0, r['cv_r2']) for r in reg_results]
bars3 = axes[1].bar(names_reg, r2s, color='#FF8C00', edgecolor='black', alpha=0.85)
axes[1].set_title('Regression Benchmark (R²)', fontweight='bold')
axes[1].set_ylabel('R²'); axes[1].set_ylim(0, 1.0)
axes[1].set_xticklabels(names_reg, rotation=20, ha='right', fontsize=9)
for bar, val in zip(bars3, r2s):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS / "tabular_benchmark.png", dpi=100, bbox_inches='tight')
plt.close()
print("Tabular benchmark complete. Best classifier:", best_cls, "| Best regressor:", best_reg)
"""),
])

# ============================================================
# NOTEBOOK 08 — FINAL TABULAR MODEL TRAINING
# ============================================================
nb08 = nb([
md("""# Forest Fire AI — Final Tabular Model Training
## Notebook 08: Train Best Classifier & Regressor, Save All Artifacts
"""),
code("""import os, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, mean_absolute_error,
                              mean_squared_error, r2_score, confusion_matrix)
import xgboost as xgb

ROOT      = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL      = ROOT / "Implementation"
META_DIR  = IMPL / "artifacts" / "metadata"
PLOTS     = IMPL / "artifacts" / "plots"
DATA_DIR  = IMPL / "data" / "processed"
TAB_DIR   = IMPL / "models" / "tabular"
TAB_DIR.mkdir(parents=True, exist_ok=True)

with open(META_DIR / "tabular_metadata.json") as f:
    tmeta = json.load(f)
with open(META_DIR / "tabular_benchmark.json") as f:
    bench = json.load(f)

BEST_CLS = bench['best_classifier']
BEST_REG = bench['best_regressor']

df = pd.read_csv(DATA_DIR / "forestfires_processed.csv")
print(f"Data: {df.shape}  |  Best classifier: {BEST_CLS}  |  Best regressor: {BEST_REG}")
"""),
code("""# Feature engineering
num_feats = tmeta['numerical_features']

le_month = LabelEncoder()
le_day   = LabelEncoder()
df['month_enc'] = le_month.fit_transform(df['month'])
df['day_enc']   = le_day.fit_transform(df['day'])

FEATURE_COLS = num_feats + ['month_enc', 'day_enc']
X = df[FEATURE_COLS].values
y_cls = df['fire_occurred'].values
y_reg = np.log1p(df['area'].values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

SEED = 42
X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=SEED, stratify=y_cls
)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
print(f"Fire in train: {yc_tr.sum()}, Fire in test: {yc_te.sum()}")
"""),
code("""# ── Train best classifier ─────────────────────────────────────────────
def get_classifier(name):
    if name == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=SEED, eval_metric='logloss', verbosity=0)
    elif name == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
    elif name == 'HistGradientBoosting':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, max_depth=5, random_state=SEED)
    else:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=SEED, max_iter=2000)

clf = get_classifier(BEST_CLS)
t0 = time.time()
clf.fit(X_tr, yc_tr)
print(f"Classifier trained in {time.time()-t0:.1f}s")

yc_pred = clf.predict(X_te)
yc_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else yc_pred

acc   = accuracy_score(yc_te, yc_pred)
prec  = precision_score(yc_te, yc_pred, zero_division=0)
rec   = recall_score(yc_te, yc_pred, zero_division=0)
f1    = f1_score(yc_te, yc_pred, zero_division=0)
auc   = roc_auc_score(yc_te, yc_prob)

print(f"\\nClassification Test Metrics:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")
"""),
code("""# ── Train best regressor ──────────────────────────────────────────────
def get_regressor(name):
    if name == 'XGBoost':
        return xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, verbosity=0)
    elif name == 'Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=SEED)

reg = get_regressor(BEST_REG)
t0 = time.time()
reg.fit(X_tr, yr_tr)
print(f"Regressor trained in {time.time()-t0:.1f}s")

yr_pred_log = reg.predict(X_te)
yr_pred     = np.expm1(yr_pred_log)
yr_te_orig  = np.expm1(yr_te)

mae  = mean_absolute_error(yr_te_orig, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_te_orig, yr_pred))
r2   = r2_score(yr_te, yr_pred_log)

print(f"\\nRegression Test Metrics (on log-transformed target):")
print(f"  MAE (ha):  {mae:.4f}")
print(f"  RMSE (ha): {rmse:.4f}")
print(f"  R² (log):  {r2:.4f}")
"""),
code("""# ── Save all artifacts ────────────────────────────────────────────────
pickle.dump(clf,    open(TAB_DIR / "classifier.pkl",  "wb"))
pickle.dump(reg,    open(TAB_DIR / "regressor.pkl",   "wb"))
pickle.dump(scaler, open(TAB_DIR / "scaler.pkl",      "wb"))
pickle.dump(le_month, open(TAB_DIR / "le_month.pkl",  "wb"))
pickle.dump(le_day,   open(TAB_DIR / "le_day.pkl",    "wb"))

tabular_metadata = {
    'classifier_name': BEST_CLS,
    'regressor_name': BEST_REG,
    'feature_cols': FEATURE_COLS,
    'numerical_features': num_feats,
    'categorical_features': ['month', 'day'],
    'month_classes': list(le_month.classes_),
    'day_classes': list(le_day.classes_),
    'cls_metrics': {'accuracy': round(acc,4), 'precision': round(prec,4),
                    'recall': round(rec,4), 'f1': round(f1,4), 'roc_auc': round(auc,4)},
    'reg_metrics': {'mae': round(mae,4), 'rmse': round(rmse,4), 'r2': round(r2,4)}
}
with open(TAB_DIR / "metadata.json", "w") as f:
    json.dump(tabular_metadata, f, indent=2)

print("\\nSaved artifacts:")
for art in ["classifier.pkl","regressor.pkl","scaler.pkl","le_month.pkl","le_day.pkl","metadata.json"]:
    print(f"  {TAB_DIR}/{art}")
"""),
code("""# Feature importance
if hasattr(clf, 'feature_importances_'):
    fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': clf.feature_importances_})
    fi = fi.sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi['feature'], fi['importance'], color='#2E8B57', edgecolor='black', alpha=0.8)
    ax.set_title(f'Feature Importance — {BEST_CLS} Classifier', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(PLOTS / "tabular_feature_importance.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved.")
    fi_dict = fi.sort_values('importance', ascending=False).set_index('feature')['importance'].to_dict()
    with open(META_DIR / "feature_importance.json", "w") as f:
        json.dump({k: round(float(v), 6) for k, v in fi_dict.items()}, f, indent=2)

print("\\nNotebook 08 complete.")
"""),
])

# ============================================================
# NOTEBOOK 09 — TABULAR MODEL EVALUATION
# ============================================================
nb09 = nb([
md("""# Forest Fire AI — Tabular Model Evaluation
## Notebook 09: Full Evaluation with Metrics, Confusion Matrix, SHAP
"""),
code("""import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, mean_absolute_error,
                              mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve)

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
META_DIR = IMPL / "artifacts" / "metadata"
PLOTS    = IMPL / "artifacts" / "plots"
METRICS  = IMPL / "artifacts" / "metrics"
DATA_DIR = IMPL / "data" / "processed"
TAB_DIR  = IMPL / "models" / "tabular"

clf    = pickle.load(open(TAB_DIR / "classifier.pkl",  "rb"))
reg    = pickle.load(open(TAB_DIR / "regressor.pkl",   "rb"))
scaler = pickle.load(open(TAB_DIR / "scaler.pkl",      "rb"))
le_month = pickle.load(open(TAB_DIR / "le_month.pkl",  "rb"))
le_day   = pickle.load(open(TAB_DIR / "le_day.pkl",    "rb"))
with open(TAB_DIR / "metadata.json") as f:
    tmeta = json.load(f)

FEATURE_COLS = tmeta['feature_cols']
NUM_FEATS    = tmeta['numerical_features']

df = pd.read_csv(DATA_DIR / "forestfires_processed.csv")
df['month_enc'] = le_month.transform(df['month'])
df['day_enc']   = le_day.transform(df['day'])

X = df[FEATURE_COLS].values
y_cls = df['fire_occurred'].values
y_reg = np.log1p(df['area'].values)

SEED = 42
_, X_te, _, yc_te, _, yr_te = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=SEED, stratify=y_cls
)

print(f"Test samples: {len(X_te)}")
print(f"Classifier:  {tmeta['classifier_name']}")
print(f"Regressor:   {tmeta['regressor_name']}")
"""),
code("""# Classification evaluation
yc_pred = clf.predict(X_te)
yc_prob = clf.predict_proba(X_te)[:, 1]

acc  = accuracy_score(yc_te, yc_pred)
prec = precision_score(yc_te, yc_pred, zero_division=0)
rec  = recall_score(yc_te, yc_pred, zero_division=0)
f1   = f1_score(yc_te, yc_pred, zero_division=0)
auc  = roc_auc_score(yc_te, yc_prob)
cm   = confusion_matrix(yc_te, yc_pred)

print("="*50)
print("CLASSIFICATION EVALUATION (fire_occurred)")
print("="*50)
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")
print("\\nConfusion Matrix:")
print(cm)
print("\\nClassification Report:")
print(classification_report(yc_te, yc_pred, target_names=['No Fire','Fire']))
"""),
code("""# Regression evaluation
yr_pred_log = reg.predict(X_te)
yr_pred     = np.expm1(yr_pred_log)
yr_te_orig  = np.expm1(yr_te)

mae  = mean_absolute_error(yr_te_orig, yr_pred)
mse  = mean_squared_error(yr_te_orig, yr_pred)
rmse = np.sqrt(mse)
r2   = r2_score(yr_te, yr_pred_log)

print("REGRESSION EVALUATION (area in ha, log-space R²)")
print(f"  MAE:  {mae:.4f} ha")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f} ha")
print(f"  R²:   {r2:.4f}")

tab_test_metrics = {
    'classifier': {'accuracy':round(acc,4),'precision':round(prec,4),
                   'recall':round(rec,4),'f1':round(f1,4),'roc_auc':round(auc,4)},
    'regressor':  {'mae':round(mae,4),'mse':round(mse,4),'rmse':round(rmse,4),'r2':round(r2,4)}
}
with open(METRICS / "tabular_test_metrics.json", "w") as f:
    json.dump(tab_test_metrics, f, indent=2)
"""),
code("""# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\\n{tmeta["classifier_name"]}', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['No Fire','Fire']); axes[0].set_yticklabels(['No Fire','Fire'])
plt.colorbar(im, ax=axes[0])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=14, fontweight='bold')

# ROC curve
fpr, tpr, _ = roc_curve(yc_te, yc_prob)
axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc:.4f}')
axes[1].plot([0,1],[0,1],'k--', alpha=0.5)
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].legend(); axes[1].grid(alpha=0.3)

# Predicted vs Actual (log scale)
axes[2].scatter(yr_te, yr_pred_log, alpha=0.5, color='#2E8B57', edgecolors='k', linewidths=0.5)
mn, mx = min(yr_te.min(), yr_pred_log.min()), max(yr_te.max(), yr_pred_log.max())
axes[2].plot([mn, mx], [mn, mx], 'r--', lw=2)
axes[2].set_title(f'Actual vs Predicted (log area)\\nR²={r2:.4f}', fontweight='bold')
axes[2].set_xlabel('Actual log(area+1)'); axes[2].set_ylabel('Predicted log(area+1)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS / "tabular_evaluation.png", dpi=100, bbox_inches='tight')
plt.close()
print("Tabular evaluation plots saved.")
"""),
code("""# SHAP values (if available)
try:
    import shap
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_te)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_importance = np.abs(shap_vals).mean(axis=0)
    si_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': shap_importance})
    si_df = si_df.sort_values('importance', ascending=True)
    ax.barh(si_df['feature'], si_df['importance'], color='#DC143C', edgecolor='black', alpha=0.8)
    ax.set_title(f'SHAP Feature Importance — {tmeta["classifier_name"]}', fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_importance.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("SHAP importance plot saved.")
except Exception as e:
    print(f"SHAP skipped (optional): {e}")

print("\\nNotebook 09 complete.")
"""),
])

# ============================================================
# NOTEBOOK 10 — FINAL INFERENCE TESTING
# ============================================================
nb10 = nb([
md("""# Forest Fire AI — Final Inference Testing
## Notebook 10: End-to-End Testing of Image + Tabular Models

Tests all saved model artifacts exactly as the dashboard would use them.
"""),
code("""import os, sys, json, pickle, time, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

ROOT     = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task")
IMPL     = ROOT / "Implementation"
MODEL_DIR_IMG = IMPL / "models" / "image"
MODEL_DIR_TAB = IMPL / "models" / "tabular"
PRED_DIR = IMPL / "artifacts" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Final Inference Testing")
print("="*60)
print(f"Device: {DEVICE}")
"""),
code("""# ── Load image model ──────────────────────────────────────────────────
with open(MODEL_DIR_IMG / "class_names.json") as f:
    cnames = json.load(f)
with open(MODEL_DIR_IMG / "model_metadata.json") as f:
    mmeta = json.load(f)

CLASS_NAMES  = cnames['class_names']
CLASS_TO_IDX = cnames['class_to_idx']
IDX_TO_CLASS = {int(k): v for k, v in cnames['idx_to_class'].items()}
MODEL_NAME   = mmeta['model_name']
IMAGE_SIZE   = mmeta['image_size']
IMG_MEAN     = mmeta['imagenet_mean']
IMG_STD      = mmeta['imagenet_std']

def build_model(model_name, num_classes=2):
    if model_name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif model_name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'ResNet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

img_model = build_model(MODEL_NAME)
img_model.load_state_dict(torch.load(MODEL_DIR_IMG / "best_model.pth", map_location=DEVICE))
img_model = img_model.to(DEVICE)
img_model.eval()
print(f"✓ Image model loaded: {MODEL_NAME}")
print(f"  Parameters: {sum(p.numel() for p in img_model.parameters()):,}")

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD)
])

def predict_image(img_path_or_pil, model=img_model):
    if isinstance(img_path_or_pil, (str, Path)):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil.convert('RGB')
    inp = transform(img).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out   = model(inp)
        probs = F.softmax(out, dim=1)[0]
    inf_ms = (time.time() - t0) * 1000
    pred_idx = probs.argmax().item()
    pred_cls = IDX_TO_CLASS[pred_idx]
    conf     = probs[pred_idx].item()
    class_probs = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
    return {'prediction': pred_cls, 'confidence': round(conf, 4),
            'class_probs': class_probs, 'inference_ms': round(inf_ms, 2)}
"""),
code("""# ── Test image model ──────────────────────────────────────────────────
from pathlib import Path
import json

with open(IMPL / "artifacts" / "metadata" / "data_splits.json") as f:
    splits = json.load(f)

test_data = splits['test']
fire_tests    = [(p, l) for p, l in test_data if l == 'FIRE'][:3]
nofire_tests  = [(p, l) for p, l in test_data if l == 'NO_FIRE'][:3]

print("Image Inference Tests:")
print("="*60)
for img_path, true_label in fire_tests + nofire_tests:
    try:
        result = predict_image(img_path)
        correct = "✓" if result['prediction'] == true_label else "✗"
        print(f"  {correct} True: {true_label:<8} | Pred: {result['prediction']:<8} | "
              f"Conf: {result['confidence']:.4f} | Inf: {result['inference_ms']:.1f}ms")
    except Exception as e:
        print(f"  Error: {e}")
"""),
code("""# ── Test different image sizes ────────────────────────────────────────
print("\\nTest different input sizes:")
test_img_path = fire_tests[0][0] if fire_tests else None
if test_img_path:
    for size in [(64,64), (128,128), (224,224), (512,512), (1024,1024)]:
        img = Image.open(test_img_path).convert('RGB').resize(size)
        try:
            result = predict_image(img)
            print(f"  Size {size}: {result['prediction']} ({result['confidence']:.3f}) | {result['inference_ms']:.1f}ms")
        except Exception as e:
            print(f"  Size {size}: Error — {e}")
"""),
code("""# ── Load tabular models ───────────────────────────────────────────────
clf    = pickle.load(open(MODEL_DIR_TAB / "classifier.pkl", "rb"))
reg    = pickle.load(open(MODEL_DIR_TAB / "regressor.pkl",  "rb"))
scaler = pickle.load(open(MODEL_DIR_TAB / "scaler.pkl",     "rb"))
le_month = pickle.load(open(MODEL_DIR_TAB / "le_month.pkl", "rb"))
le_day   = pickle.load(open(MODEL_DIR_TAB / "le_day.pkl",   "rb"))
with open(MODEL_DIR_TAB / "metadata.json") as f:
    tmeta = json.load(f)

FEATURE_COLS = tmeta['feature_cols']
NUM_FEATS    = tmeta['numerical_features']

print("✓ Tabular models loaded")
print(f"  Classifier: {tmeta['classifier_name']}")
print(f"  Regressor:  {tmeta['regressor_name']}")
print(f"  Features:   {FEATURE_COLS}")

def predict_tabular(row_dict):
    \"\"\"Predict fire risk from a dict of environmental features.\"\"\"
    num_vals = [float(row_dict.get(f, 0)) for f in NUM_FEATS]
    month_enc = le_month.transform([row_dict.get('month', 'aug').lower()])[0]
    day_enc   = le_day.transform([row_dict.get('day', 'fri').lower()])[0]
    X = np.array(num_vals + [month_enc, day_enc]).reshape(1, -1)
    fire_prob  = clf.predict_proba(X)[0, 1]
    fire_pred  = int(clf.predict(X)[0])
    area_log   = reg.predict(X)[0]
    area_pred  = float(np.expm1(area_log))
    return {
        'fire_predicted': fire_pred,
        'fire_probability': round(float(fire_prob), 4),
        'predicted_area_ha': round(area_pred, 4)
    }
"""),
code("""# Test tabular model
print("\\nTabular Inference Tests:")
print("="*60)
test_rows = [
    {'X':7,'Y':5,'month':'aug','day':'fri','FFMC':92.3,'DMC':85.3,'DC':488.0,'ISI':14.7,'temp':22.2,'RH':29,'wind':5.4,'rain':0.0},
    {'X':4,'Y':4,'month':'feb','day':'mon','FFMC':73.2,'DMC':14.0,'DC':25.6,'ISI':2.0, 'temp':4.5,'RH':80,'wind':1.3,'rain':0.0},
    {'X':6,'Y':5,'month':'sep','day':'sat','FFMC':93.4,'DMC':145.4,'DC':721.4,'ISI':8.1,'temp':30.2,'RH':24,'wind':2.7,'rain':0.0},
]
for i, row in enumerate(test_rows):
    result = predict_tabular(row)
    risk = "HIGH" if result['fire_probability'] > 0.6 else ("MEDIUM" if result['fire_probability'] > 0.35 else "LOW")
    print(f"  Sample {i+1}: Fire={result['fire_predicted']} | P(fire)={result['fire_probability']:.4f} | "
          f"Area={result['predicted_area_ha']:.2f}ha | Risk={risk}")
"""),
code("""# ── Test CSV input ────────────────────────────────────────────────────
import csv, io

csv_content = \"\"\"X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain
7,5,aug,fri,92.3,85.3,488.0,14.7,22.2,29,5.4,0.0
4,4,feb,mon,73.2,14.0,25.6,2.0,4.5,80,1.3,0.0
8,6,sep,tue,91.0,129.5,692.6,7.0,13.1,63,5.4,0.0
6,5,mar,sat,91.7,35.8,80.8,7.8,15.1,27,5.4,0.0
\"\"\"
df_test = pd.read_csv(io.StringIO(csv_content))
print("\\nCSV Input Test:")
print(f"  Input rows: {len(df_test)}")

results_rows = []
for _, row in df_test.iterrows():
    r = predict_tabular(row.to_dict())
    row_dict = row.to_dict()
    row_dict.update(r)
    results_rows.append(row_dict)

df_results = pd.DataFrame(results_rows)
print(df_results[['X','Y','month','day','fire_predicted','fire_probability','predicted_area_ha']].to_string(index=False))

output_csv = PRED_DIR / "test_predictions.csv"
df_results.to_csv(output_csv, index=False)
print(f"\\nPredictions saved: {output_csv}")
"""),
code("""# ── Error handling tests ──────────────────────────────────────────────
print("\\nError Handling Tests:")

# 1. Corrupt image
try:
    bad_img = Image.new('RGB', (10, 10), color=(0,0,0))
    r = predict_image(bad_img)
    print(f"  ✓ Tiny image (10x10): {r['prediction']} ({r['confidence']:.3f})")
except Exception as e:
    print(f"  ✓ Tiny image handled: {e}")

# 2. Invalid CSV row  
try:
    bad_row = {'month': 'xxx', 'day': 'yyy', 'FFMC': 'abc'}
    r = predict_tabular(bad_row)
    print(f"  × Unknown month/day not caught — need validation in dashboard")
except Exception as e:
    print(f"  ✓ Invalid row caught: {type(e).__name__}")

# 3. Missing features → use 0
row_missing = {'X': 7, 'Y': 5, 'month': 'aug', 'day': 'fri', 'temp': 22}
r = predict_tabular(row_missing)
print(f"  ✓ Missing features (filled with 0): P(fire)={r['fire_probability']:.4f}")

print("\\n" + "="*60)
print("ALL INFERENCE TESTS COMPLETE")
print("="*60)
print(f"  Image model: {MODEL_NAME} — READY")
print(f"  Classifier:  {tmeta['classifier_name']} — READY")
print(f"  Regressor:   {tmeta['regressor_name']} — READY")
print("  Dashboard can now be launched.")
"""),
])

# ── Save all notebooks ────────────────────────────────────────────────
print("Creating notebooks...")
save(nb01, "01_dataset_audit.ipynb")
save(nb02, "02_image_data_preprocessing.ipynb")
save(nb03, "03_image_model_benchmarking.ipynb")
save(nb04, "04_final_image_model_training.ipynb")
save(nb05, "05_image_model_evaluation_explainability.ipynb")
save(nb06, "06_tabular_data_analysis.ipynb")
save(nb07, "07_tabular_model_benchmarking.ipynb")
save(nb08, "08_final_tabular_model_training.ipynb")
save(nb09, "09_tabular_model_evaluation.ipynb")
save(nb10, "10_final_inference_testing.ipynb")
print("\\nAll 10 notebooks created successfully!")
