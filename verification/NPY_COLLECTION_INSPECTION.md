# Inspection of the uploaded `images.zip` / `masks.zip` (measured, not assumed)
## images.zip (49 MB, 296 MB uncompressed, zip integrity OK)
| Item | Result |
|---|---|
| Files | 1,280 `.npy` + 1 `brain_image.nii.gz` |
| Name pattern | all 1,280 match `BraTS20_Training_<id>_slice<n>.npy`; 0 unreadable, 0 non-finite, 0 all-zero, 0 duplicate contents |
| Shape / dtype | all `(4, 240, 240)` uint8 |
| Patients | **360** distinct (IDs 001-369). Absent IDs: 002, 066, 089, 100, 104, 223, 267, 288, 355 |
| Slices per patient | 1: 43 · 2: 63 · 3: 86 · 4: 76 · 5: 43 · 6: 24 · 7: 15 · 8: 6 · 9: 3 · 10: 1 patients (sparse random sample, not full volumes) |
| Slice indices | 0-144, all 145 values occur (quartiles 0 / 36 / 70 / 107 / 144) |
| Brain content | median 19.6% non-zero pixels; 1,126 of 1,280 slices have > 5% |
| `brain_image.nii.gz` | the same 64^3 uniform-noise file as before (SHA-256 70347f8b...) - NOT MRI |
Intensity facts: **every one of the 4 channels of every slice has min = 0 and max = 255** -> each channel of each slice was individually rescaled to 0-255 (a per-volume scaling would reach 255 on only one slice per volume). Mean in-brain correlation between stored channels (0,1,2,3): 0-1: 0.55, 0-2: -0.39, 0-3: 0.41, 1-2: -0.32, 1-3: 0.23, 2-3: 0.05. This is only a hint (channel 2 differs from the others, channels 0 and 1 are most alike); it does NOT identify the modalities.
## masks.zip (1,145 bytes)
Contains only `brain_mask.nii.gz`: a 64x64x64 binary volume, 7,153 voxels, a sphere of radius 12 centred at (32,32,32) - a synthetic placeholder matching the noise volume, **not a tumour mask**. **There are no `.npy` mask slices** -> 0 of 1,280 images have a mask (see `npy_inventory.csv`, column `mask_file_available`).
## Is the data sufficient for the preprocessing probe?
**No - not yet.** Images are plentiful (360 patients, 1,280 slices) but the probe needs masks. Masks can be regenerated without guessing from the BraTS NIfTI `*_seg` of each patient with `scripts/match_npy_to_nifti.py`, which first proves (by pixel matching) how your slices were cut from the NIfTI volumes.
