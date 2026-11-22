# Limitations
* **Checkpoint provenance unknown**: the file is a bare `state_dict` (no metadata, epoch, patient list or config). Training/validation patients are unknown, so independent test performance **cannot be verified**; no such claim is made.
* **Training preprocessing unknown**: normalization, channel order and class mapping in `model_config.yaml` are unverified defaults. Wrong preprocessing silently degrades predictions.
* Weight/architecture compatibility was verified (strict load, 0 missing/unexpected/mismatched keys) – this says nothing about accuracy.
* BraTS is skull-stripped, co-registered, 1 mm isotropic, from a limited set of institutions; behaviour on other data is unknown. No external validation.
* 2-D slice-wise inference ignores through-plane context. Segmentation errors propagate into volumes and radiomics; radiomics are sensitive to bin width, resampling and normalisation, and arbitrary MRI intensities make absolute values non-comparable across scanners. Not full IBSI compliance.
* No clinical validation, no regulatory clearance; outputs must not guide care. Not HIPAA-compliant; no formal security audit.
* Authentication/multi-user access control **not implemented** (single-user local academic deployment; see `security.md`). 
* Classification, Grad-CAM++, TreeSHAP and survival prognosis are **not implemented** (no genuine trained models; see `explainability.md`).
* Front-end libraries load from CDNs (internet needed in browser).
* Processing time is hardware dependent (see benchmark; measured on a 1-vCPU sandbox only). Concurrency: jobs are serialised; behaviour under heavy multi-user load not tested.
* GPU path is implemented but was **not tested** (no GPU available in the build environment).
