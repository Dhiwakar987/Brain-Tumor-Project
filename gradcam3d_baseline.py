from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
DATA_DIR = PROJECT / "prepared_3d_patches"
CHECKPOINT = PROJECT / "model" / "weights" / "unet3d_baseline.pth"
OUTPUT = PROJECT / "model" / "explanations" / "gradcam3d_baseline.npz"
CROP = 32
TARGET_CLASS = 3

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

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

with (DATA_DIR / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

val_records = [r for r in records if r["split"] == "val"]
if not val_records:
    raise RuntimeError("No validation patches found.")

record = val_records[0]
with np.load(Path(record["file"])) as data:
    image = data["image"].astype(np.float32)

if image.shape != (4, 64, 64, 64):
    raise RuntimeError(f"Unexpected image shape: {image.shape}")

start = (image.shape[1] - CROP) // 2
image = image[:, start:start+CROP, start:start+CROP, start:start+CROP]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyUNet3D().to(device)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

captured = {}

def capture_activation(module, inputs, output):
    output.retain_grad()
    captured["activation"] = output

hook = model.mid.register_forward_hook(capture_activation)

x = torch.from_numpy(image.copy()).unsqueeze(0).to(device)
x.requires_grad_(True)

model.zero_grad(set_to_none=True)
logits = model(x)
score = logits[:, TARGET_CLASS].mean()
score.backward()

activation = captured["activation"]
gradients = activation.grad
weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
cam = F.interpolate(cam, size=(CROP, CROP, CROP), mode="trilinear", align_corners=False)
cam = cam[0, 0].detach().cpu().numpy()

cam_min, cam_max = float(cam.min()), float(cam.max())
if cam_max > cam_min:
    cam = (cam - cam_min) / (cam_max - cam_min)
else:
    cam = np.zeros_like(cam, dtype=np.float32)

hook.remove()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT,
    gradcam=cam.astype(np.float32),
    case=record["case"],
    target_class=TARGET_CLASS,
)

print("=== 3D GRAD-CAM DIAGNOSTIC ===")
print("Checkpoint loaded:", CHECKPOINT)
print("Validation case:", record["case"])
print("Target class:", TARGET_CLASS)
print("Grad-CAM shape:", cam.shape)
print("Grad-CAM range:", float(cam.min()), "to", float(cam.max()))
print("Saved:", OUTPUT)
print("Diagnostic only: smoke-test checkpoint; explanation is not validated.")

