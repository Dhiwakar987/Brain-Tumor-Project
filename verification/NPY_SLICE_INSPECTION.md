# Training-format slice inspection (`BraTS20_Training_001_slice8.npy`, uploaded)
Loaded with `allow_pickle=False` (no code execution).
| Property | Measured |
|---|---|
| File size / SHA-256 (first 16) | 230,528 bytes / `a54577b74ae066a4` |
| Shape, dtype | **(4, 240, 240), uint8** – 4 channels, one 240×240 axial slice |
| Value range | 0 … 255 (211 distinct values), all finite |
| Non-zero pixels | 0.48% (slice 8 is at the very bottom of the brain; mostly background) |
| Channel meaning / order | **UNKNOWN** – not stored in the file or the filename |
| Patient / slice | BraTS20_Training_001, axial slice index 8 (from the filename; the slice axis is assumed to be the last NIfTI axis) |

## What this tells us about training preprocessing
* Training slices are stored as **8-bit (0–255) 4-channel images** – they were intensity-rescaled before saving. This is incompatible with my earlier guess "z-score of raw intensities" being stored as-is.
* **Checkpoint cross-check (measured):** the first BatchNorm layer stores training-time statistics. `bn1.running_var` median = 0.0216. For this slice, `conv1` output variance is 271 if the network were fed raw 0–255 values (~12,000× too large), 0.0042 if divided by 255, and 0.0106 if z-scored. So the model was **not** trained on raw 0–255 inputs; it was trained on inputs of roughly unit scale (÷255 or standardised). This cannot separate ÷255 from z-scoring and says nothing about channel order or label mapping; the slice is also mostly empty, so the numbers are only indicative.
* Not determined: whether scaling was per volume or per slice, channel order, mask format, class mapping. `model/model_config.yaml` flags therefore stay `false`.

## How to settle it (uses your own data)
```powershell
python scripts\probe_preprocessing.py --images "data\images" --masks "data\masks" --limit 40
```
It runs the real checkpoint on your training-format slices under each candidate preprocessing and all 24 channel orders and compares with your masks (whole-tumour Dice is independent of the label mapping). Results: `verification\PREPROCESSING_PROBE.md`. Caveat: these slices were probably in the training set, so use it to choose the configuration, never to report accuracy.
