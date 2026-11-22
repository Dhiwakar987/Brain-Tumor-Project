from pathlib import Path
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
DATA_DIR = PROJECT / "prepared_3d_patches"
MANIFEST = DATA_DIR / "manifest.csv"
CHECKPOINT = PROJECT / "model" / "weights" / "unet3d_baseline.pth"

CROP = 32
EPOCHS = 3
BATCH_SIZE = 1
NUM_CLASSES = 4

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

with MANIFEST.open("r", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

train_records = [r for r in records if r["split"] == "train"]
val_records = [r for r in records if r["split"] == "val"]

if not train_records or not val_records:
    raise RuntimeError("Manifest must contain both train and validation patches.")

for row in train_records + val_records:
    if not Path(row["file"]).exists():
        raise FileNotFoundError(f"Patch file missing: {row['file']}")

print(f"Training patches: {len(train_records)}")
print(f"Validation patches: {len(val_records)}")
print(f"Epochs: {EPOCHS}; crop: {CROP}^3")

class PatchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        with np.load(Path(self.rows[idx]["file"])) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.int64)

        if image.ndim != 4 or image.shape[0] != 4:
            raise RuntimeError(f"Expected 4 MRI channels; got image shape={image.shape}")
        if label.ndim != 3 or tuple(image.shape[1:]) != tuple(label.shape):
            raise RuntimeError(
                f"Image/label spatial mismatch: image={image.shape}, label={label.shape}"
            )
        if len(set(label.shape)) != 1 or label.shape[0] < CROP:
            raise RuntimeError(f"Expected a cubic patch at least {CROP} voxels wide; got {label.shape}")

        start = (label.shape[0] - CROP) // 2
        image = image[:, start:start+CROP, start:start+CROP, start:start+CROP]
        label = label[start:start+CROP, start:start+CROP, start:start+CROP]

        if label.min() < 0 or label.max() >= NUM_CLASSES:
            raise RuntimeError(f"Unexpected label values: {np.unique(label)}")

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())

def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class TinyUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = block(4, 4)
        self.enc2 = block(4, 8)
        self.mid = block(8, 16)
        self.pool = nn.MaxPool3d(2)
        self.dec2 = block(16 + 8, 8)
        self.dec1 = block(8 + 4, 4)
        self.out = nn.Conv3d(4, NUM_CLASSES, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        m = self.mid(self.pool(b))
        x = F.interpolate(m, size=b.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec2(torch.cat([x, b], dim=1))
        x = F.interpolate(x, size=a.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec1(torch.cat([x, a], dim=1))
        return self.out(x)

def combined_loss(logits, target):
    # Lower background weight reduces its dominance over tumor classes.
    weights = torch.tensor([0.2, 1.0, 1.0, 1.0], device=logits.device)
    ce = F.cross_entropy(logits, target, weight=weights)

    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probs * target_onehot).sum(dims)
    denominator = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)

    # Foreground classes only: edema, non-enhancing, enhancing.
    dice_loss = 1.0 - dice[1:].mean()
    return ce + dice_loss

def validate(model, loader, device):
    model.eval()
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    pred_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    true_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_loss = 0.0

    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            total_loss += combined_loss(logits, label).item()
            pred = logits.argmax(dim=1)

            for cls in range(NUM_CLASSES):
                intersections[cls] += ((pred == cls) & (label == cls)).sum().item()
                pred_counts[cls] += (pred == cls).sum().item()
                true_counts[cls] += (label == cls).sum().item()

    dice_scores = []
    for cls in range(1, NUM_CLASSES):
        denom = pred_counts[cls] + true_counts[cls]
        score = (2.0 * intersections[cls] / denom) if denom else float("nan")
        dice_scores.append(score)

    mean_dice = float(np.nanmean(dice_scores))
    return total_loss / max(len(loader), 1), dice_scores, mean_dice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_loader = DataLoader(
    PatchDataset(train_records), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    PatchDataset(val_records), batch_size=1,
    shuffle=False, num_workers=0
)

model = TinyUNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_dice = -1.0
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    for step, (image, label) in enumerate(train_loader, 1):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = combined_loss(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 25 == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"step {step}/{len(train_loader)} "
                f"loss={loss.item():.4f}",
                flush=True
            )

    val_loss, class_dice, mean_dice = validate(model, val_loader, device)
    elapsed = time.time() - start_time

    print(f"\nEpoch {epoch} train_loss={running_loss / len(train_loader):.4f}")
    print(f"Validation loss={val_loss:.4f}")
    print(f"Dice class 1 (edema)={class_dice[0]:.4f}")
    print(f"Dice class 2 (non-enhancing)={class_dice[1]:.4f}")
    print(f"Dice class 3 (enhancing)={class_dice[2]:.4f}")
    print(f"Mean foreground Dice={mean_dice:.4f}")
    print(f"Epoch time={elapsed / 60:.1f} minutes", flush=True)

    if np.isfinite(mean_dice) and mean_dice > best_dice:
        best_dice = mean_dice
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_channels": 4,
            "classes": NUM_CLASSES,
            "crop_size": CROP,
            "epoch": epoch,
            "validation_mean_foreground_dice": best_dice,
            "note": "Small 3D baseline; patch-based and not clinically validated."
        }, CHECKPOINT)
        print("Saved best checkpoint:", CHECKPOINT, flush=True)

print("\nTraining finished.")
print("Best validation mean foreground Dice:", best_dice)
print("Best checkpoint:", CHECKPOINT)
print("This is a patch-based research baseline, not a clinical model.")

