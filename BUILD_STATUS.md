# BUILD_STATUS
Environment: Linux, Python 3.12.3, 1 vCPU, 4 GB RAM, **no GPU**, torch 2.15.0 (CPU path), smp 0.6.0, PyRadiomics 3.1.1.dev111 (GitHub source build).
## Completed modules (implemented)
config · model/{model_loader,preprocessing,inference,checkpoint_validation} · segmentation/{nifti_io,prediction,regions,metrics} · radiomics_analysis/{extractor,feature_export,settings.yaml} · database/{db,schema.sql} · services/{analysis_service,file_service} · reports/pdf_generator · visualization/{mri_viewer,plots} · Flask app (all required routes + extras) · templates/static · scripts (verify_checkpoint, check_inputs, backup_restore, benchmark) · experiments (split, segmentation+paired radiomics, figures) · docs.
## Not implemented
User authentication/ownership · Grad-CAM · model comparison (E7) · automatic retention cleanup · a separate `report_service.py` (logic is `AnalysisService.generate_report`) · GPU testing · background-job resource monitoring beyond benchmark.
## Tests executed (this workspace)
* `pytest` with real checkpoint path: **55 passed, 0 failed, 0 skipped** (includes 2 integration tests).
* `pytest` without checkpoint: **53 passed, 2 skipped** (integration).
* All volumes in unit tests are SYNTHETIC; "checkpoint" fixture = random weights.
## Verification evidence
| Item | Evidence |
|---|---|
| Checkpoint file inspected, strict load into smp.Unet(resnet34,4,4) | `verification/MODEL_VERIFICATION_REPORT.md`, `checkpoint_inspection.json` (278 tensors, 0 missing/unexpected/mismatch; 24,439,940 params; SHA-256 a15086d5…f5e32b) |
| Forward pass of real weights | Synthetic random-noise input only (NOT real MRI) |
| Preprocessing/label mapping vs training | NOT VERIFIED (training code not supplied) |
| Real BraTS inference/experiments | NOT RUN (no data) |
| Flask start, /health, pages, upload→segment→evaluate→report with CSRF | Live HTTP smoke test using real checkpoint on synthetic volumes |
| Benchmark | `verification/benchmark_results.json` (synthetic volumes, 1 vCPU) |
## Errors found and fixed during build
* NIfTI temp file `*.partial` rejected by nibabel (extension-based writer) → renamed to `partial_<name>.nii.gz`; regression covered by test.
* Test assertion on empty history page corrected.
* PyRadiomics PyPI sdist failed to build → installed from GitHub source (documented).

# v1.1 upgrade (this session)
**Added:** design system + all templates restyled (dashboard, upload, results, radiomics, evaluation, report, history, overview); tabs for 2D viewer / three-panel / 3D / explainability status; `segmentation/measurements.py`; `visualization/mesh3d.py`; endpoints `compare-slice`, `mesh3d`, `measurements`, `export`; CSV/XLSX/JSON export; PDF page numbers + measurements + three-panel figure; `scripts/inspect_mri.py`.
**Tests:** `pytest` (no checkpoint): 60 passed, 2 skipped. With real checkpoint path: 62 passed, 0 failed. All data synthetic.
**Live HTTP check (real checkpoint, synthetic 64x64x16 volumes):** all pages, new endpoints and static files returned 200; 0 tracebacks in server log.
**Front-end:** JS files and inline scripts pass `node --check`; **NOT run in a browser** (no browser available) – Plotly viewers, drag-drop, charts and responsive layout are untested visually. **No screenshots could be produced.**
**Phase 3:** no `brain_image.nii` / modality / mask files exist in the workspace – see `verification/MRI_FILE_VERIFICATION.md`.
**Not implemented (no genuine model/data):** classification, Grad-CAM++, TreeSHAP, survival.

# v2.0 (this session)
**Inputs inspected:** `brain_image_nii.gz` (64³ float32, uniform random noise – statistically verified, NOT MRI, modality unknown), `best_unet_brats_3.pth` (byte-identical to `best_unet_brats.pth`), re-uploaded project zip (identical to v1.1, no data), screenshot of `data/images` (1,280 per-slice .npy, not uploaded). No T1/T1ce/T2/FLAIR NIfTI, no segmentation mask, no BraTS case was supplied.
**Fixes:** `/tmp` paths in tests/benchmark -> portable temp dirs; NIfTI memmap -> copy (Windows file locks); noise-volume warning; environment checker.
**Added:** Model performance / Classification / Explainable AI pages, docs (DATASET_SETUP, MODEL_DETAILS, PROJECT_ARCHITECTURE), requirements-windows.txt.
**Tests:** without checkpoint 68 passed / 0 failed / 2 skipped; with checkpoint 70 passed / 0 failed. All on Linux; **not run on Windows** (the Windows-specific fixes are untested there). One transient failure during development was caused by the sandbox disk being full (leftover pytest temp files), not by the code; after cleanup everything passed.
**Still not done:** real-MRI inference, preprocessing/label-mapping verification, classification, Grad-CAM++/TreeSHAP, survival, browser/visual testing, screenshots.

# v2.0.1 (this session)
Dataset **not** received: `brain_image_nii_1.gz` is byte-identical to the earlier noise file (SHA-256 70347f8b…); no BraTS patient files were uploaded, so dataset verification, real-patient segmentation, genuine metrics, real-data radiomics and viewer checks are **NOT RUN**.
Added only what is essential and data-independent: `scripts/verify_dataset.py` (+2 tests); `run_segmentation_experiment.py` now works without a split file (labelled non-independent) and was smoke-tested end to end with the real checkpoint on a tiny SYNTHETIC patient (plumbing only; its metrics are meaningless).

# v2.0.2 (this session)
Uploaded `BraTS20_Training_001_slice8.npy` inspected safely: (4,240,240) uint8 0-255. Checkpoint BN statistics (bn1.running_var median 0.0216) show the model was NOT trained on raw 0-255 inputs; ÷255 vs z-score, channel order, mask format and label mapping remain UNVERIFIED. Added `scripts/probe_preprocessing.py` (+3 tests; plumbing tested on synthetic data only, never run on real masks because none were supplied) and `verification/NPY_SLICE_INSPECTION.md`. Config defaults unchanged; verification flags still false.

# v2.0.3 (this session)
**Received this turn:** nothing new. Only ONE slice exists on my side (`BraTS20_Training_001_slice8.npy`); no masks, no training code/notebook, no NIfTI. Channel order, preprocessing choice and label mapping are therefore **still UNVERIFIED**; config flags remain false.
**Tested here (Linux):** real checkpoint forward pass on the real slice8 (÷255): output (1,4,256,256), finite, predicted all-background (slice 8 is the brain's bottom; no mask, so no accuracy claim). Probe CLI smoke test with the real checkpoint on synthetic noise slices: runs; verdict logic correctly reports NOT DECIDED.
**Fixed (Windows hazards found by code review, not run on Windows):** .env UTF-8 BOM + quotes; explicit UTF-8 for cache files; SQLite WAL disabled automatically inside OneDrive/Dropbox paths (override with SQLITE_JOURNAL_MODE); probe image/mask size check; probe verdicts + too-few-slices warning; eval() replaced by ast.literal_eval; *.npy and data/ git-ignored. Added docs/NEXT_STEPS_WINDOWS.md.

# v2.0.4 (this session) - image collection received
**Received & inspected:** `images.zip` (1,280 real BraTS .npy slices, 360 patients, all (4,240,240) uint8, every channel min 0 / max 255 => per-slice per-channel rescaling) and `masks.zip` (only a synthetic 64^3 sphere `brain_mask.nii.gz`; NO .npy masks). Full facts: `verification/NPY_COLLECTION_INSPECTION.md`, inventory `verification/npy_inventory.csv`.
**Sufficient for the probe? NO** - zero image-mask pairs. Channel order, orientation, label mapping and training normalisation remain UNVERIFIED; config flags unchanged (false).
**Added:** `scripts/match_npy_to_nifti.py` (proves channel order/axis/orientation/scaling by pixel-matching slices to their source NIfTI; writes masks only after a verified, cross-patient-consistent match; tested on SYNTHETIC data only - never run on real NIfTI); invertible `slice_orientation` option in the inference pipeline (default identity = unchanged behaviour; round-trip + pipeline tests). Bugs found by tests while building: tuple-assignment NameError, runner-up logic, tie handling.
**Not tested:** anything on real NIfTI/masks; Windows.
## Dataset probe update (2026-11-22)
- `_output_.zip` contains 50,706 image slices and 50,706 mask slices.
- Extracted image folder contains slices for 368 patient IDs.
- Probe ran on 40 paired slices: stored channel order (0,1,2,3) was the clear WT-Dice winner (0.899; runner-up 0.755). This supports the stored order diagnostically, but does not verify modality names.
- Normalization remains inconclusive: div255 and div_slice_max tied on the probe. Keep modality names and normalization marked UNVERIFIED.
- Probe is a training-data diagnostic, not an independent accuracy evaluation.
