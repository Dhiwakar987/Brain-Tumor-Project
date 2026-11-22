# Dataset setup (BraTS 2020)
## What is needed
Per patient: **T1, T1ce, T2, FLAIR** volumes of the same patient, co-registered on the same 240×240×155 grid at 1 mm, plus the expert segmentation (`seg`) with labels **0** background, **1** necrotic/non-enhancing core, **2** edema, **4** enhancing tumour. BraTS 2020 provides exactly this (`.nii.gz`), already skull-stripped.
## Where to get it (you must do this yourself – nothing was downloaded for you)
1. Official challenge page (University of Pennsylvania CBICA): <https://www.med.upenn.edu/cbica/brats2020/data.html>. Access requires registration on the CBICA Image Processing Portal and acceptance of the data-usage terms; the page explains the steps. Check the current licence/terms there before use.
2. Community mirrors exist (e.g. on Kaggle), but verify the licence, completeness (369 training cases) and integrity yourself; the official source is preferred for a final-year report.
3. Training set: 369 cases (`MICCAI_BraTS2020_TrainingData`), with `name_mapping.csv` (contains tumour grade HGG/LGG) and `survival_info.csv`.
## Windows steps (PowerShell)
```powershell
# after downloading the archive to Downloads:
Expand-Archive -Path "$env:USERPROFILE\Downloads\MICCAI_BraTS2020_TrainingData.zip" -DestinationPath "C:\data\brats2020"
# or use 7-Zip for .tar/.tar.gz archives
```
## Folder layout used by this project
```
C:\data\brats2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\
    BraTS20_Training_001_t1.nii.gz   BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz   BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_seg.nii.gz
```
Web app: upload the five files of ONE patient on *New MRI analysis* (the four sequences + `_seg` as ground truth). Batch scripts: `$env:BRATS_DATASET_PATH="C:\data\brats2020\MICCAI_BraTS2020_TrainingData"`.
Check any files first: `python scripts\inspect_mri.py --folder C:\data\brats2020\...\BraTS20_Training_001` (3-D/4-D, spacing, orientation, labels, alignment, noise warning).
## Your current files
See `verification/MRI_FILE_VERIFICATION.md`: the uploaded `brain_image` file is random noise, and the per-slice `.npy` files cannot be used by the app. Use the original NIfTI cases above.
## Leakage rule
Split **by patient**, never by slice. Your 1,280 per-slice `.npy` files come from several patients; slices of the same patient must never be spread over train and test.

## Verify your dataset locally (no upload needed) – do this first
```powershell
python scripts\verify_dataset.py "C:\data\brats2020\MICCAI_BraTS2020_TrainingData" --limit 20
```
Writes `verification\DATASET_VERIFICATION.md` and `dataset_verification.csv`: per patient, whether T1/T1ce/T2/FLAIR/SEG exist and are readable, same shape/affine, valid seg labels (0,1,2,4), noise warning, and any file whose modality cannot be identified (reported as UNKNOWN, never guessed). Paste the table to the assistant for review.
## Run the real model on real patients (batch, no web upload needed)
```powershell
$env:CHECKPOINT_PATH = "model\weights\best_unet_brats.pth"
python experiments\run_segmentation_experiment.py --dataset "C:\data\brats2020\MICCAI_BraTS2020_TrainingData" --radiomics --limit 3 --out results\first_check
python experiments\make_figures.py --results results\first_check
```
Without `--split-csv` all folders are used and results are labelled *NOT VERIFIED AS INDEPENDENT TEST*. Outputs: `per_patient_metrics.csv` (Dice/IoU/precision/recall/specificity/HD95 per region), `radiomics_paired.csv`, `summary.json`, `run_metadata.json`.
