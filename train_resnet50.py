"""
Forest Sentinel — ResNet50 Image Classification Training
==========================================================
Trains ResNet50 (pretrained on ImageNet) on the fire/no-fire dataset.
Optimised for CPU-only training on Intel i5-1240P / 16 GB RAM.
Saves metrics alongside YOLO26 for benchmark comparison.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("FOREST SENTINEL — ResNet50 IMAGE MODEL TRAINING")
print("=" * 70)
print(f"PyTorch:  {torch.__version__}")
print(f"Device:   cpu (Intel i5-1240P)")
print()

# ─── Paths ───────────────────────────────────────────────────────────────────
IMPL        = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task/Implementation")
META_DIR    = IMPL / "artifacts" / "metadata"
METRICS_DIR = IMPL / "artifacts" / "metrics"
PLOTS_DIR   = IMPL / "artifacts" / "plots"
IMG_DIR     = IMPL / "models" / "image"
IMG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data splits
with open(META_DIR / "data_splits.json") as f:
    splits = json.load(f)

with open(META_DIR / "class_weights.json") as f:
    cw_data = json.load(f)

# ─── Config ──────────────────────────────────────────────────────────────────
SELECTED_MODEL = "ResNet50"
IMAGE_SIZE     = 224
BATCH_SIZE     = 16
EPOCHS         = 20
PATIENCE       = 5
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
CLASS_NAMES    = ['FIRE', 'NO_FIRE']
CLASS_TO_IDX   = {'FIRE': 0, 'NO_FIRE': 1}
IDX_TO_CLASS   = {0: 'FIRE', 1: 'NO_FIRE'}
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

print(f"Selected model: {SELECTED_MODEL}")
print(f"Image size:     {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Max epochs:     {EPOCHS}")
print(f"Early stopping: {PATIENCE}")
print()

# ─── Dataset ─────────────────────────────────────────────────────────────────
class FireDataset(Dataset):
    def __init__(self, items, transform, class_to_idx=None):
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.samples = []
        for item in items:
            if isinstance(item, list):
                path, cls = item[0], item[1]
                label = self.class_to_idx.get(cls, 0)
            elif isinstance(item, dict):
                path  = item.get("path", "")
                label = item.get("label", 0)
            else:
                continue
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 64, 0))
        return self.transform(img), label

train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.2),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomGrayscale(p=0.02),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
val_tf = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_items = splits["train"]
val_items   = splits["val"]
test_items  = splits["test"]

print("Building datasets...")

train_ds = FireDataset(train_items, train_tf, CLASS_TO_IDX)
val_ds   = FireDataset(val_items,   val_tf,   CLASS_TO_IDX)
test_ds  = FireDataset(test_items,  val_tf,   CLASS_TO_IDX)

N_TRAIN = len(train_ds.samples)
N_VAL   = len(val_ds.samples)
N_TEST  = len(test_ds.samples)

print(f"Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─── Model ───────────────────────────────────────────────────────────────────
def build_resnet50(nc=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze early layers (layers 1-3), unfreeze layer4 + fc
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
    # Replace the final fully-connected layer
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, nc)
    )
    return model

print(f"\nBuilding {SELECTED_MODEL}...")
model = build_resnet50(nc=2)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Total params:     {total:,}")
print(f"  Trainable params: {trainable:,}")

# ─── Loss & Optimizer ────────────────────────────────────────────────────────
# Use class weights from the dataset analysis
class_weights_list = cw_data.get("weights", [1.0, 1.0])
if not isinstance(class_weights_list, list) or len(class_weights_list) < 2:
    # Fallback: compute from class_weights dict
    cw_dict = cw_data.get("class_weights", {"FIRE": 1.0, "NO_FIRE": 1.0})
    class_weights_list = [cw_dict.get("FIRE", 1.0), cw_dict.get("NO_FIRE", 1.0)]

weight_tensor = torch.tensor(class_weights_list, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─── Training Loop ────────────────────────────────────────────────────────────
def evaluate(loader, mdl):
    mdl.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out   = mdl(imgs)
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)
    acc  = accuracy_score(labels_arr, preds_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_arr, preds_arr,
                                                         average='weighted', zero_division=0)
    try:    auc = roc_auc_score(labels_arr, probs_arr)
    except: auc = 0.5
    return acc, prec, rec, f1, auc, labels_arr, preds_arr

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1    = 0.0
no_improve = 0
best_epoch = 0

print(f"\n{'='*70}")
print(f"TRAINING: {SELECTED_MODEL}  (max {EPOCHS} epochs, patience={PATIENCE})")
print(f"{'='*70}")

t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_i, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

        if (batch_i + 1) % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS}  Batch {batch_i+1}/{len(train_loader)}  "
                  f"Loss={running_loss/(BATCH_SIZE*(batch_i+1)):.4f}", end='\r')

    scheduler.step()

    epoch_loss = running_loss / N_TRAIN

    # Validation
    val_acc, val_prec, val_rec, val_f1, val_auc, _, _ = evaluate(val_loader, model)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs)
            val_loss += criterion(out, labels).item() * imgs.size(0)
    val_loss /= N_VAL

    elapsed = time.time() - t0

    history["train_loss"].append(round(epoch_loss, 4))
    history["val_loss"].append(round(val_loss, 4))
    history["val_acc"].append(round(val_acc, 4))
    history["val_f1"].append(round(val_f1, 4))

    print(f"  Epoch {epoch:2d}/{EPOCHS} | "
          f"train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | "
          f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
          f"val_auc={val_auc:.4f} | {elapsed:.0f}s")

    # Checkpoint if best
    if val_f1 > best_f1:
        best_f1    = val_f1
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), IMG_DIR / "best_model_resnet50.pth")
        print(f"  ** New best model saved (F1={best_f1:.4f}) **")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered (no improvement for {PATIENCE} epochs)")
            break

train_time = time.time() - t_start
print(f"\nTraining complete in {train_time/60:.1f} min. Best epoch: {best_epoch}, Best Val F1: {best_f1:.4f}")

# ─── Save training history ─────────────────────────────────────────────────
with open(META_DIR / "resnet50_training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Training history saved.")

# ─── Plot Training Curves ─────────────────────────────────────────────────
epochs_ran = list(range(1, len(history["train_loss"]) + 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes: ax.set_facecolor('#161b22')

axes[0].plot(epochs_ran, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
axes[0].plot(epochs_ran, history['val_loss'],   'r-s', markersize=4, label='Val Loss')
axes[0].set_title('Loss Curves', color='white', fontweight='bold')
axes[0].set_xlabel('Epoch', color='#8b949e'); axes[0].set_ylabel('Loss', color='#8b949e')
axes[0].legend(facecolor='#161b22', edgecolor='#21262d', labelcolor='white')
axes[0].tick_params(colors='#8b949e')
for sp in axes[0].spines.values(): sp.set_color('#21262d')
axes[0].grid(alpha=0.15, color='white')

axes[1].plot(epochs_ran, history['val_acc'], 'g-^', markersize=4, label='Val Accuracy')
axes[1].plot(epochs_ran, history['val_f1'],  'm-D', markersize=4, label='Val F1')
axes[1].set_title('Validation Metrics', color='white', fontweight='bold')
axes[1].set_xlabel('Epoch', color='#8b949e'); axes[1].set_ylabel('Score', color='#8b949e')
axes[1].legend(facecolor='#161b22', edgecolor='#21262d', labelcolor='white')
axes[1].tick_params(colors='#8b949e')
for sp in axes[1].spines.values(): sp.set_color('#21262d')
axes[1].grid(alpha=0.15, color='white')
axes[1].set_ylim(0, 1)

plt.suptitle(f'ResNet50 Training — Best Val F1={best_f1:.4f}',
             color='white', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "resnet50_training_curves.png", dpi=120, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Training curves plot saved.")

# ─── Test Evaluation ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TEST SET EVALUATION — ResNet50")
print(f"{'='*70}")

# Load best model
model.load_state_dict(torch.load(IMG_DIR / "best_model_resnet50.pth", map_location='cpu'))
model.eval()

t0 = time.time()
test_acc, test_prec, test_rec, test_f1, test_auc, y_true, y_pred = evaluate(test_loader, model)
inf_ms = (time.time() - t0) / N_TEST * 1000

print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1:        {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")
print(f"  Inference: {inf_ms:.2f}ms/image")

cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# ─── Save Test Metrics ────────────────────────────────────────────────────
test_metrics = {
    "accuracy":     round(float(test_acc), 4),
    "precision":    round(float(test_prec), 4),
    "recall":       round(float(test_rec), 4),
    "f1":           round(float(test_f1), 4),
    "roc_auc":      round(float(test_auc), 4),
    "inference_ms": round(float(inf_ms), 2),
    "confusion_matrix": cm.tolist(),
    "model_name":   SELECTED_MODEL,
    "best_epoch":   best_epoch,
    "best_val_f1":  round(float(best_f1), 4),
    "test_samples": len(test_items),
    "training_time_min": round(train_time / 60, 1),
}
with open(METRICS_DIR / "resnet50_test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print("Test metrics saved.")

# ─── Update Benchmark Results ─────────────────────────────────────────────
bench_path = META_DIR / "benchmark_results.json"
if bench_path.exists():
    with open(bench_path) as f:
        benchmarks = json.load(f)
else:
    benchmarks = []

# Remove old ResNet18 and ResNet50 entries, add ResNet50
benchmarks = [b for b in benchmarks if b.get("model") not in ("ResNet18", "ResNet50")]
benchmarks.append({
    "model":        SELECTED_MODEL,
    "accuracy":     round(float(test_acc), 4),
    "precision":    round(float(test_prec), 4),
    "recall":       round(float(test_rec), 4),
    "f1":           round(float(test_f1), 4),
    "inference_ms": round(float(inf_ms), 2),
})
with open(bench_path, "w") as f:
    json.dump(benchmarks, f, indent=2)
print("Benchmark results updated (ResNet18 → ResNet50).")

# ─── Confusion Matrix Plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
im = ax.imshow(cm, cmap='Greens')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(CLASS_NAMES, color='white')
ax.set_yticklabels(CLASS_NAMES, color='white')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] < cm.max() * 0.6 else 'black',
                fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', color='#8b949e')
ax.set_ylabel('True', color='#8b949e')
ax.set_title(f'Confusion Matrix — {SELECTED_MODEL}\nTest Acc={test_acc:.4f}, F1={test_f1:.4f}',
             color='white', fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "resnet50_confusion_matrix.png", dpi=120, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Confusion matrix plot saved.")

print()
print("=" * 70)
print("ResNet50 IMAGE MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"  Model:      {SELECTED_MODEL}")
print(f"  Test Acc:   {test_acc*100:.2f}%")
print(f"  Test F1:    {test_f1:.4f}")
print(f"  Test AUC:   {test_auc:.4f}")
print(f"  Train Time: {train_time/60:.1f} min")
print(f"  Saved to:   {IMG_DIR / 'best_model_resnet50.pth'}")
print("=" * 70)
