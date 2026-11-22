# MRI file verification – files actually uploaded (Nov 2026)
| # | Exact filename | Modality | Patient ID | Shape | Voxel spacing (mm) | Segmentation mask? | Valid / readable |
|---|---|---|---|---|---|---|---|
| 1 | `brain_image_nii.gz` (gzip; inner file is a NIfTI-1, 1,048,928 bytes) | **UNKNOWN** – header description empty, no filename token (t1/t1ce/t2/flair), no sequence info | **Not identifiable** (no ID in name or header) | 64 × 64 × 64 (3-D, float32) | 1.0 × 1.0 × 1.0 | No – continuous float values 4.7e-6 … 1.0 (a mask would hold labels 0/1/2/4) | **Readable: yes. Valid anatomical MRI: NO** (see below) |
| 2 | `best_unet_brats.pth`, `best_unet_brats_3.pth` | – model weights, not images (identical files: SHA-256 `a15086d5…f5e32b`) | – | – | – | – | Loads strictly into smp.Unet(resnet34, 4 in, 4 classes) |

## Why file 1 is not usable as MRI
Statistical tests on the actual voxel data (not guesses):
* Every voxel is non-zero and the volume has **no zero background** (real skull-stripped BraTS volumes are ~60–80% zero).
* Values are in (0, 1) with mean 0.4997 and SD 0.2888 – the SD of a Uniform(0,1) variable is 0.2887.
* Kolmogorov–Smirnov test against Uniform(0,1): statistic 0.0017, **p = 0.44** (cannot be distinguished from uniform random noise); the 10 decile bins hold 25,937–26,357 voxels each (expected 26,214).
* Correlation between neighbouring voxels along x, y, z: **−0.002, −0.0001, −0.003** (anatomical MRI is typically > 0.8).
* Rendered axial/coronal/sagittal slices show pure salt-and-pepper noise.
It is therefore almost certainly a **randomly generated placeholder volume**, not a brain scan. The application now flags such files at upload ("looks like random noise").
Other header facts: orientation LPS-flipped identity affine (`L,P,S`), qform/sform codes 1.

## Answers
* **All four modalities per patient available?** NO – there is no patient with any identified modality; one file with unknown modality only.
* **Ground-truth masks available?** NO.
* **Missing files:** for each patient `*_t1`, `*_t1ce`, `*_t2`, `*_flair` (`.nii`/`.nii.gz`) and `*_seg`.
* **Compatible with the U-Net?** The checkpoint expects **4 input channels** (T1, T1ce, T2, FLAIR; order unverified) – one 3-D file cannot supply them, and substituting/duplicating channels is refused by design. The 64³ size is technically paddable (multiple of 32), but the content is noise, so any prediction would be meaningless. **Not compatible / not usable.**

## Also seen (screenshot only – files NOT uploaded)
`data/images/` listing: a file `brain_image.nii` shown as "Compressed Archive" (918 KB – matches the gzip uploaded above, i.e. it is gzip data without a `.gz` extension) and **1,280 files named `BraTS20_Training_<id>_slice<n>.npy`, ~226 KB each**. These look like per-slice exports of BraTS 2020 training cases (consistent with ~240×240 float32, but 4×120×120 float32 would be the same size) – shape, channel layout and normalisation cannot be determined without opening them. They are **not NIfTI**, cannot be inspected or used by this app, and no modality or segmentation can be assumed from the names.
