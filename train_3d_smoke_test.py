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
CHECKPOINT = PROJECT / "model" / "weights" / "unet3d_smoke_test.pth"
CROP = 32
TRAIN_LIMIT = 2
VAL_LIMIT = 1

print("Waiting for the 3D patch-preparation process to finish...")
while True:
    if MANIFEST.exists():
        try:
            with MANIFEST.open("r", newline="", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
            if len(records) == 484 and all(Path(r["file"]).exists() for r in records):
                break
        except (PermissionError, OSError, csv.Error):
            pass
    time.sleep(10)

train_records = [r for r in records if r["split"] == "train"][:TRAIN_LIMIT]
val_records = [r for r in records if r["split"] == "val"][:VAL_LIMIT]
if not train_records or not val_records:
    raise RuntimeError("Could not find both train and validation patches in manifest.")

class PatchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        with np.load(self.rows[idx]["file"]) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.int64)

        # Center-crop to 32³ to keep CPU memory use low.
        starts = [(n - CROP) // 2 for n in label.shape]
        x, y, z = starts
        image = image[:, x:x+CROP, y:y+CROP, z:z+CROP]
        label = label[x:x+CROP, y:y+CROP, z:z+CROP]

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
        self.out = nn.Conv3d(4, 4, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        m = self.mid(self.pool(b))
        x = F.interpolate(m, size=b.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec2(torch.cat([x, b], dim=1))
        x = F.interpolate(x, size=a.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec1(torch.cat([x, a], dim=1))
        return self.out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(f"Training patches: {len(train_records)}; validation patches: {len(val_records)}")

train_loader = DataLoader(PatchDataset(train_records), batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(PatchDataset(val_records), batch_size=1, shuffle=False, num_workers=0)

model = TinyUNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

model.train()
for step, (image, label) in enumerate(train_loader, 1):
    image, label = image.to(device), label.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(image), label)
    loss.backward()
    optimizer.step()
    print(f"Train step {step}/{len(train_loader)} loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    for step, (image, label) in enumerate(val_loader, 1):
        loss = loss_fn(model(image.to(device)), label.to(device))
        print(f"Validation step {step}/{len(val_loader)} loss={loss.item():.4f}")

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "input_channels": 4,
    "classes": 4,
    "crop_size": CROP,
    "note": "Tiny 3D smoke test only; not a trained or clinically validated model."
}, CHECKPOINT)

print("\n=== 3D SMOKE TEST COMPLETE ===")
print("Checkpoint:", CHECKPOINT)
print("This confirms the code can train and save a tiny 3D model; it does not establish model accuracy.")
