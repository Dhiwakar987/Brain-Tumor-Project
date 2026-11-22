# Viva preparation (answers reflect what is implemented; [PENDING] = depends on work you still must do)
1. **Contribution beyond a notebook?** A validated, reproducible web workflow: input QC, checkpoint verification, configurable preprocessing, regions/volumes, radiomics with stored settings, evaluation with explicit policy, DB history, PDF/JSON export, tests. No algorithmic novelty is claimed.
2. **Why T1/T1ce/T2/FLAIR?** BraTS standard: T1ce highlights enhancing tumour/BBB disruption, T1 anatomy and necrosis context, T2 and FLAIR edema.
3. **Why U-Net + ResNet34?** Skip connections preserve localisation; a ResNet34 encoder is a compact, well-known backbone (24.4 M parameters total as measured). Alternatives not compared [PENDING: E7].
4. **Checkpoint compatibility?** `verify_checkpoint.py`: instantiate smp.Unet(resnet34, 4 in, 4 out), strict load → 278 tensors, 0 missing/unexpected/mismatched (see MODEL_VERIFICATION_REPORT.md). Compatibility ≠ accuracy.
5. **Preprocessing matches training?** Not verified – training code wasn't available. It is configurable and flagged unverified. [PENDING: compare with your training code and set the flags.]
6. **Label mapping?** Config `class_mapping` (default 3→4), BraTS WT={1,2,4}, TC={1,4}, ET={4}. Mapping is unverified [PENDING].
7. **Leakage prevention?** Patient-wise splits only (`make_patient_split.py`); checkpoint's own training patients unknown, so independence can't be asserted [PENDING].
8. **Metrics?** Dice/IoU (overlap), precision/recall/specificity (error types), HD95 (boundary error, robust to outliers).
9. **HD95?** 95th percentile of symmetric surface distances in **mm** using voxel spacing.
10. **Radiomics on predicted mask?** The ground truth is unavailable in practice; features from predictions show what a deployed pipeline would produce.
11. **Predicted vs reference radiomics?** `--radiomics` experiment / comparison table: same scan and settings, differences per feature (abs, relative %). [PENDING: run on real data]
12. **Reproducibility?** Settings + PyRadiomics version stored per run; config snapshot, checkpoint SHA-256, input hashes, versions in metadata JSON/PDF.
13. **Wrong segmentation?** Warnings for implausible masks, visual review, no auto-correction; radiomics inherit the error.
14. **No ground truth?** Evaluation refused with an explanation; no accuracy shown.
15. **SQLite contents?** Metadata, hashes, relative file names, results, jobs, audit; MRI volumes are files (size, DB bloat, tool compatibility).
16. **Protection of images?** Whitelist+sniffing, random folders, no path exposure, CSRF, local-only default; no auth/encryption at rest – stated limitation.
17. **Limitations?** See `limitations.md`.
18. **Experiments supporting conclusions?** Software tests (synthetic) and a benchmark on synthetic volumes are done. **No BraTS accuracy results exist yet** – run the experiment scripts. [PENDING]
