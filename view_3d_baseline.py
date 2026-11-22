from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
PRED_FILE = PROJECT / "model" / "predictions" / "smoke_test_prediction.npz"
CAM_FILE = PROJECT / "model" / "explanations" / "gradcam3d_baseline.npz"
OUT_FILE = PROJECT / "model" / "explanations" / "baseline_comparison.png"

if not PRED_FILE.exists():
    raise FileNotFoundError(PRED_FILE)
if not CAM_FILE.exists():
    raise FileNotFoundError(CAM_FILE)

with np.load(PRED_FILE) as d:
    prediction = d["prediction"]
    label = d["label"]
    case = str(d["case"])

with np.load(CAM_FILE) as d:
    gradcam = d["gradcam"]

if prediction.shape != label.shape or prediction.shape != gradcam.shape:
    raise RuntimeError(
        f"Shape mismatch: prediction={prediction.shape}, "
        f"label={label.shape}, Grad-CAM={gradcam.shape}"
    )

z = prediction.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(label[:, :, z], interpolation="nearest")
axes[0].set_title("Ground truth")
axes[1].imshow(prediction[:, :, z], interpolation="nearest", vmin=0, vmax=3)
axes[1].set_title("Smoke-test prediction")
axes[2].imshow(gradcam[:, :, z], interpolation="nearest", vmin=0, vmax=1)
axes[2].set_title("Diagnostic Grad-CAM")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"{case} â€” middle axial slice (z={z})")
fig.tight_layout()

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print("Saved comparison image:", OUT_FILE)
plt.show()


