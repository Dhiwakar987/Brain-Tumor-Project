# Remaining steps (do in this order)
Your `data\images` collection (1,280 slices, 360 patients) is inspected - see `verification\NPY_COLLECTION_INSPECTION.md`. **No mask slices exist yet.**
## Step 1 - prove how the slices were made and regenerate the masks (needs your BraTS NIfTI folder)
```powershell
cd C:\path\to\NeuroRadiomics
.\.venv\Scripts\Activate.ps1
python scripts\match_npy_to_nifti.py --nifti-root "C:\data\MICCAI_BraTS2020_TrainingData" --images "data\images" --write-masks "data\masks"
```
`--nifti-root` must contain one folder per patient (`BraTS20_Training_001\BraTS20_Training_001_t1.nii(.gz)`, `_t1ce`, `_t2`, `_flair`, `_seg`). The tool matches your stored slices pixel-for-pixel against the NIfTI volumes and reports, from data: **which stored channel is T1 / T1ce / T2 / FLAIR, the slice axis, the in-plane orientation and the intensity scaling**, confirms the same mapping on 4 more patients, and only then writes mask slices (original BraTS labels 0/1/2/4) into `data\masks`. If anything is ambiguous it prints NOT DETERMINED and writes nothing. Read `verification\NPY_NIFTI_MATCH.md`.
## Step 2 - configure what was proven
In `model\model_config.yaml`: `modality_order` = the channel order from Step 1 *converted to the order of your files* (stored channel i = modality X means the model expects X in channel i), `preprocessing.slice_orientation` = the orientation found, and `normalization: minmax_slice` if the scaling was per-slice (this app then divides to 0-1 exactly like stored/255). Slice axis must be 2 (the pipeline supports only axial slicing along the last axis).
## Step 3 - label mapping and final check with the probe
```powershell
python scripts\probe_preprocessing.py --images "data\images" --masks "data\masks" --limit 60
```
Read the Verdict and the confusion matrix (clean diagonal = mapping 4->3 is right). Then set the `verification:` flags in `model_config.yaml` to true for what was supported.
## Step 4 - real patients
`python scripts\verify_dataset.py <BraTS folder> --limit 20`; `python app.py` and upload one patient's 5 NIfTI files; `python experiments\run_segmentation_experiment.py --dataset <BraTS folder> --radiomics --limit 3 --out results\first_check`.
