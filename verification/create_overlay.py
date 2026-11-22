
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the main project folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model_loader import load_model
from model.inference import predict_slice

name = "BraTS20_Training_001_slice100.npy"

image = np.load(Path("data/brats2d/brats2d/images") / name).astype(np.float32)
mask = np.load(Path("data/brats2d/brats2d/masks") / name)

cfg = yaml.safe_load(open("model/model_config.yaml", encoding="utf-8"))

image = image / 255.0
h, w = image.shape[1:]
ph, pw = (-h) % 32, (-w) % 32
top, left = ph // 2, pw // 2

padded = np.pad(
    image,
    ((0, 0), (top, ph-top), (left, pw-left))
)

model, device = load_model(
    "model/weights/best_unet_brats.pth",
    cfg,
    device_preference="cpu"
)

pred = predict_slice(model, padded, device)
pred = pred[top:top+h, left:left+w]

mri = image[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(mri, cmap="gray")
axes[0].imshow(
    np.ma.masked_where(mask == 0, mask > 0),
    cmap="autumn",
    alpha=0.65
)
axes[0].set_title("Ground Truth Overlay")

axes[1].imshow(mri, cmap="gray")
axes[1].imshow(
    np.ma.masked_where(pred == 0, pred > 0),
    cmap="autumn",
    alpha=0.65
)
axes[1].set_title("U-Net Prediction Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path(__file__).resolve().parent / "2d_prediction_overlay.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(str(output), dpi=160, bbox_inches="tight")
plt.close()

print("Overlay saved:", output)

print("Overlay saved:", output.resolve())
