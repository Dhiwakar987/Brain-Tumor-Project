from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
DATA_DIR = PROJECT / "prepared_3d_patches"
CHECKPOINT = PROJECT / "model" / "weights" / "unet3d_baseline.pth"
OUTPUT = PROJECT / "model" / "predictions" / "smoke_test_prediction.npz"
CROP = 32

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

with (DATA_DIR / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

val_records = [r for r in records if r["split"] == "val"]
if not val_records:
    raise RuntimeError("No validation patches found in manifest.")

record = val_records[0]
patch_path = Path(record["file"])

with np.load(patch_path) as data:
    image = data["image"].astype(np.float32)
    label = data["label"].astype(np.int64)

if image.shape != (4, 64, 64, 64) or label.shape != (64, 64, 64):
    raise RuntimeError(f"Unexpected patch shapes: image={image.shape}, label={label.shape}")

# Use the same center crop as the smoke-test training script.
start = (image.shape[1] - CROP) // 2
image = image[:, start:start+CROP, start:start+CROP, start:start+CROP]
label = label[start:start+CROP, start:start+CROP, start:start+CROP]

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
model = TinyUNet3D().to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

input_tensor = torch.from_numpy(image.copy()).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT,
    prediction=prediction,
    label=label.astype(np.uint8),
    case=record["case"]
)

print("\n=== 3D CHECKPOINT INFERENCE CHECK ===")
print("Checkpoint loaded:", CHECKPOINT)
print("Validation case:", record["case"])
print("Input shape:", tuple(input_tensor.shape))
print("Prediction shape:", prediction.shape)
print("Prediction class counts:", dict(zip(*np.unique(prediction, return_counts=True))))
print("Ground-truth class counts:", dict(zip(*np.unique(label, return_counts=True))))
print("Saved prediction:", OUTPUT)
print("Diagnostic only: this smoke-test model is not accuracy-validated.")


