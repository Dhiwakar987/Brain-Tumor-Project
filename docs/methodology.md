# Methodology
## 1. Established methods used (not contributions of this project)
U-Net (Ronneberger 2015), ResNet34 encoder, `segmentation_models_pytorch`, BraTS label scheme and WT/TC/ET definitions, PyRadiomics, Dice/IoU/HD95.
## 2. Engineering/integration contribution (implemented)
Validated multimodal ingestion, checkpoint verification tooling, configurable preprocessing, region/volume computation, radiomics with stored settings, predicted-vs-reference radiomics comparison, evaluation with explicit empty-mask policy, history/DB, PDF + JSON reproducibility export, job persistence, backup/consistency tooling, test suite.
## 3. Pipeline
1. **Validation** – identical shape, affine (atol 1e-3), orientation codes, valid positive spacing, finite and non-zero voxels. Misaligned scans are *rejected*, never resampled.
2. **Preprocessing** – per `model/model_config.yaml`: channel order, normalization (`zscore_nonzero | minmax_volume | minmax_slice | none`), optional percentile clip, zero-padding H/W to a multiple of 32 (cropped back after inference). **Whether this equals training preprocessing is UNVERIFIED** (training code not supplied).
3. **Inference** – 2-D U-Net over axial slices (`slice_axis: 2`), `model.eval()`, `torch.inference_mode()`, argmax. CUDA if available.
4. **Label mapping** – class→BraTS via `class_mapping` (default 3→4, UNVERIFIED). Unknown class indices raise an error.
5. **Regions** – WT {1,2,4}, TC {1,4}, ET {4}; volume mL = voxels × voxel mm³ / 1000.
6. **QC** – heuristic warnings (empty mask, tiny/huge volume, fragmentation, ET > WT). They never modify predictions and are not a validated quality score. Human review flag: not reviewed / reviewed / requires attention (not clinical approval).
7. **Radiomics** – PyRadiomics, original image only, 3-D, `binWidth 25`, no resampling, no normalisation (all in `settings.yaml`, stored with every run, together with the PyRadiomics version, regions, modalities, source). Shape features are mask-only and stored once per region with modality `mask_only`. Geometry is validated; empty regions and extraction errors are recorded as failures rather than crashing. IBSI: terminology/feature definitions follow PyRadiomics' IBSI-referenced implementation; **full IBSI compliance is not claimed or assessed** (no IBSI phantom benchmark run; bin width unjustified for arbitrary MRI intensities).
8. **Evaluation** – Dice, IoU, precision, recall, specificity, HD95 (mm, physical spacing, pooled symmetric 95th percentile of surface distances). Policy: undefined → N/A (never silently 0/1); one empty mask → Dice=IoU=0.

## 4. Experimental protocol (scripts provided; results NOT RUN here)
* **Patient-wise split only** – `experiments/make_patient_split.py` (seeded; all modalities/slices of a patient together). Test set untouched during any tuning.
* **Provenance** – checkpoint training patients are unknown → results from `run_segmentation_experiment.py` are labelled *NOT VERIFIED AS INDEPENDENT TEST* unless you pass `--independent-confirmed` after establishing independence. If you cannot establish it, retrain with a fresh patient-wise split (not provided) or evaluate on an external cohort.
* **E1–E4** (baseline, WT/TC/ET, per-patient distributions with mean/SD/median/IQR/n/undefined counts, per-volume & per-slice timing): `run_segmentation_experiment.py` → `per_patient_metrics.csv`, `summary.json`, `failures.csv`, `run_metadata.json`; `make_figures.py` → plots/tables.
* **E5** CPU vs GPU: run the script with `--device cpu` and `--device cuda` on the same machine; compare only on the same hardware.
* **E6** robustness: automated tests cover missing/corrupt/misaligned/incompatible inputs (synthetic).
* **E7** model comparison: NOT implemented.
* **Paired radiomics** (`--radiomics`): same scan, same settings, ground-truth vs predicted mask → `radiomics_paired.csv` (abs and relative differences). Differences reflect segmentation differences, not clinical significance.
* **System tests**: `scripts/benchmark.py` (stage times, RSS, CPU time, output size).
Never fill result tables by hand; every figure must be generated from saved CSV/JSON.
