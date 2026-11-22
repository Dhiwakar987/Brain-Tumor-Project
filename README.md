# NeuroRadiomics – Brain Tumor Segmentation and Radiomics Analysis (v2.0)

> **Research prototype – not a medical device.** No clinical validation, no regulatory approval, not HIPAA-compliant. Use de-identified / public (BraTS) data only.

Working title option (if your supervisor allows): *Design and Development of an AI-Powered Web-Based Brain Tumor Segmentation and Radiomics Analysis System Using Multimodal MRI.*

## Your manual to-do list (nothing below can be done without your files/PC)
1. Install and run the Windows commands below; send me any error text.
2. `python scripts\verify_dataset.py <your BraTS folder> --limit 20` and share `verification\DATASET_VERIFICATION.md`.
3. Run `experiments\run_segmentation_experiment.py ... --radiomics --limit 3` on real patients and inspect the overlays in the web app (upload one patient's 5 files).
4. **Run the preprocessing probe on your own training-format slices** (decisive, ~2 min on CPU): `python scripts\probe_preprocessing.py --images "data\images" --masks "data\masks" --limit 40`, then send me `verification\PREPROCESSING_PROBE.md`. (Mask folder/name may differ – use `--mask-template`.) Also provide your training notebook/script if you still have it, (+ two `.npy` samples and masks) so preprocessing, channel order and label mapping can be verified; then set the flags in `model\model_config.yaml`. Until then segmentation quality may be wrong.
5. Decide for the report: no classifier / Grad-CAM++ / TreeSHAP / survival exist (shown as unavailable). Train one on a patient-wise split only if your supervisor requires it.
6. Take your own screenshots for the report (none were produced during development).

## v2.0 summary
* New pages: **Model performance**, **Tumour classification** and **Explainable AI** (the latter two honestly report "unavailable" – no genuine models exist).
* Upload validation now **warns when a volume looks like random noise** (found in the uploaded `brain_image` file – see `verification/MRI_FILE_VERIFICATION.md`).
* **Windows fixes:** tests/benchmark use portable temp folders (no `/tmp`); NIfTI data is copied, not memory-mapped (so files can be deleted on Windows); `scripts\check_environment.py` diagnoses NumPy/PyRadiomics problems; `requirements-windows.txt`.
* Docs: `docs/PROJECT_ARCHITECTURE.md`, `docs/DATASET_SETUP.md`, `docs/MODEL_DETAILS.md`.

## Windows quick start (PowerShell, Python 3.11 recommended)
```powershell
cd C:\Users\numes\OneDrive\Documents\NeuroRadiomics
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-windows.txt
python scripts\check_environment.py     # must print OK for every line
copy .env.example .env                   # then edit SECRET_KEY
mkdir model\weights -ErrorAction SilentlyContinue
copy "C:\path\to\best_unet_brats.pth" model\weights\best_unet_brats.pth
python scripts\verify_checkpoint.py
python -m pytest -q
python app.py                            # open http://127.0.0.1:5000
```
Tip: keep the project outside OneDrive "Files On-Demand" folders if you see file-lock or sync errors (OneDrive can lock files while syncing).

## Preprocessing in simple words
1. Open each sequence with NiBabel and check they sit on the same grid (same shape, same affine, same orientation, spacing > 0, no NaN/Inf, not all zero).
2. Normalise each sequence separately (default: z-score over the non-zero brain voxels – **a guess until you confirm it matches your training**).
3. Stack the four sequences as 4 channels in the configured order (T1, T1ce, T2, FLAIR).
4. Pad height/width to a multiple of 32 (needed by the ResNet34 U-Net), run the model slice by slice, crop the padding away, stack the slices into a 3-D volume and map class numbers to BraTS labels.
The output is saved with the original affine and header, so measurements and overlays keep the correct millimetre geometry.

## What's new in v1.1 (UI/UX and analysis upgrade of the existing Flask app)
* **New design system** – navy sidebar, light canvas, teal accents; dashboard with real-data charts (activity, completed/pending, WT volume per analysis, radiomics runs), model status panel, prominent *New MRI analysis*.
* **Upload page** – four equal modality cards + ground-truth card, drag-and-drop, per-card validation state (format/NIfTI header), progress, server validation messages.
* **Results viewer** – tabs: **2D viewer** (axial/coronal/sagittal, slice slider + wheel, zoom in/out/reset, drag-pan, opacity, WT/TC/ET, MRI/prediction/GT source, anatomical axis labels), **three-panel comparison** (Original MRI | Ground truth | AI prediction, same slice), **3D brain** (marching-cubes surfaces from the real MRI + real predicted labels, in mm), **Explainability** (states honestly what is unavailable).
* **Measurements** – volume (mm³, mL), voxel count, surface area, sphericity, max 3-D diameter, surface/volume, bounding-box extent – from header spacing.
* **Exports** – full analysis as CSV, Excel, JSON; PDF now has page numbers, measurements and the three-panel figure.
* `scripts/inspect_mri.py` – verifies any NIfTI (3-D/4-D, spacing, orientation, labels, mutual alignment).
* Unchanged: Flask architecture, U-Net/ResNet34 loading (strict), preprocessing config, PyRadiomics, SQLite, jobs, tests.
**Not available (not fabricated):** classification, Grad-CAM++, TreeSHAP, survival prognosis – no genuine trained models exist in this project.
The old standalone `extract_features.py` is kept as `scripts/legacy_extract_features.py` (it uses default PyRadiomics settings and a hard-coded path; the app uses `radiomics_analysis/` with documented settings).

## What it does
Multimodal MRI upload (T1, T1ce, T2, FLAIR; optional ground-truth) → strict validation → U-Net/ResNet34 slice-wise segmentation → WT/TC/ET regions + volumes → PyRadiomics (7 feature families, per region & modality) → Dice/IoU/precision/recall/specificity/HD95 (only with a reference mask) → interactive viewer → PDF report, CSV/Excel, JSON reproducibility metadata → SQLite history.

**Contribution statement (proposed, not a novelty claim):** engineering integration and evaluation workflow of established methods (U-Net, ResNet34, multimodal MRI, PyRadiomics). No "first-of-its-kind" claim is made without a literature review.

## Architecture
See `docs/architecture.md` (Mermaid diagram). Layers: Flask routes (`app.py`) → services (`services/`) → ML/IO modules (`model/`, `segmentation/`, `radiomics_analysis/`) → SQLite (`database/`) + files (`uploads/`, `outputs/`). Heavy jobs run in a single-worker background queue with persisted job states.

> Deviation from the requested layout: the radiomics package is named **`radiomics_analysis/`** (not `radiomics/`) because a local folder called `radiomics` would shadow the real PyRadiomics library.

## Folder structure
```
app.py config.py requirements*.txt .env.example pytest.ini
model/          model_loader.py preprocessing.py inference.py checkpoint_validation.py model_config.yaml weights/
segmentation/   prediction.py regions.py metrics.py nifti_io.py
radiomics_analysis/ extractor.py settings.yaml feature_export.py
database/       db.py schema.sql
services/       analysis_service.py file_service.py   (report_service: report generation lives in analysis_service.generate_report)
reports/        pdf_generator.py
visualization/  mri_viewer.py plots.py
templates/ static/{css,js}   uploads/ outputs/
tests/          pytest suite (synthetic data)
scripts/        verify_checkpoint.py check_inputs.py backup_restore.py benchmark.py
experiments/    make_patient_split.py run_segmentation_experiment.py make_figures.py
verification/   MODEL_VERIFICATION_REPORT.md checkpoint_inspection.json INPUT_AVAILABILITY.md benchmark_results.json
docs/           architecture methodology installation user_manual limitations viva_preparation security demonstration_checklist explainability
BUILD_STATUS.md
```

## Install (summary – details in `docs/installation.md`)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
# 1) PyTorch first (pick CPU or CUDA command at https://pytorch.org/get-started/locally/), e.g. CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2) the rest
pip install -r requirements-dev.txt
cp .env.example .env        # Windows: copy .env.example .env   then edit SECRET_KEY
```
**Windows:** use `copy .env.example .env` and `.venv\Scripts\activate`; CPU works (GPU optional via the CUDA PyTorch build).
**Place the checkpoint at** `model/weights/best_unet_brats.pth` (or set `CHECKPOINT_PATH` in `.env`).

## Run
```bash
python scripts/verify_checkpoint.py      # writes verification/MODEL_VERIFICATION_REPORT.md
python app.py                            # http://127.0.0.1:5000   (health: /health)
python -m pytest -q                      # unit tests (synthetic data); integration tests skip without the checkpoint
python -m pytest -m integration -q       # real-checkpoint tests
```

## BEFORE you trust any result – you must do these manually
1. Open `model/model_config.yaml` and make **normalization, channel order, class mapping** match your *training* code. Defaults (z-score over non-zero voxels, order T1/T1ce/T2/FLAIR, classes 0/1/2/3→labels 0/1/2/4) are **unverified guesses**. Then set the `verification:` flags to `true` only when confirmed.
2. Establish which BraTS patients trained/tuned the checkpoint. Until then, any metric is "NOT VERIFIED AS INDEPENDENT TEST".
3. Run on real BraTS cases and inspect overlays visually.

## BraTS 2020 setup
Download BraTS 2020 yourself; keep it outside the project. Expected layout: `<root>/BraTS20_Training_001/BraTS20_Training_001_{t1,t1ce,t2,flair,seg}.nii.gz`. In the web UI upload the four modalities (+ `_seg` for evaluation). For batch experiments set `BRATS_DATASET_PATH` and see `docs/methodology.md` §Experiments.

## Using each page
Home (overview) · Dashboard (DB-driven stats, model info) · New analysis (upload with progress + validation) · Results (run segmentation, axial/coronal/sagittal viewer, WT/TC/ET selector, opacity, legend, review flag, downloads) · Radiomics (run per source/region/modality, filter, histogram, CSV/Excel, predicted-vs-GT comparison) · Evaluation (needs GT; metrics table/chart/policy) · Report (generate/download PDF, metadata JSON) · History (open, PDF, delete with confirmation) · Methodology.

## Operations
Backup/restore/consistency: `python scripts/backup_restore.py backup|verify|restore|check` (see `docs/installation.md`). Benchmark: `python scripts/benchmark.py`.

## Using the new features
* **2D viewer:** choose modality/view/region/source; wheel or slider = slice; drag = pan; magnifier buttons = zoom; opacity slider.
* **Three-panel comparison:** upload a ground-truth mask with the case; tab *Three-panel comparison* shows the same slice in all panels. Without GT it says the comparison is unavailable.
* **3D:** tab *3D brain* – drag to rotate, scroll to zoom, toggle brain/edema, change brain-surface modality. Needs internet for Plotly CDN.
* **Export:** *Reports & export* page → CSV / Excel / JSON, PDF, reproducibility metadata.
* **Check your files first:** `python scripts/inspect_mri.py <files>`.

## Troubleshooting
* *"Checkpoint file not found"* – fix `CHECKPOINT_PATH`.  *"not compatible with configured architecture"* – run `verify_checkpoint.py`, inspect the mismatch list.
* *Plots blank* – Plotly/Bootstrap/Chart.js load from CDNs; you need internet access in the browser (or vendor the files into `static/`).
* *PyRadiomics won't install* – see `docs/installation.md`.
* *Slow on CPU* – ~0.2 s/slice measured on a 1-vCPU sandbox (see `verification/benchmark_results.json`); use a GPU for speed.

## Known limitations
See `docs/limitations.md`. Highlights: unverified preprocessing/label mapping, unknown checkpoint provenance, no authentication (single-user local use), no clinical validation, CDN-dependent front-end, 2-D slice-wise inference only.
