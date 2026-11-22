# Model details
## Segmentation (available)
* **U-Net** = encoder–decoder CNN: the encoder shrinks the image while extracting features, the decoder expands it back to a per-pixel class map, **skip connections** carry fine detail across. Here the encoder is a **ResNet34** (residual blocks) from `segmentation_models_pytorch`.
* Verified facts (`verification/MODEL_VERIFICATION_REPORT.md`): bare `state_dict`, 278 tensors, strict load into `smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=4)` with 0 missing / 0 unexpected / 0 shape-mismatched keys, 24,439,940 parameters. `best_unet_brats.pth` and `best_unet_brats_3.pth` are byte-identical.
* **Not verifiable from the file:** training data, patient split, epochs, loss, normalisation, channel order, label mapping, 2-D slice vs 3-D training. The checkpoint has no metadata and the training code was not supplied. The model is **2-D** (a 4-channel slice in, 4-class map out) – the app runs it slice-by-slice and stacks the results into a 3-D volume.
* **New evidence (uploaded `BraTS20_Training_001_slice8.npy`):** training slices are `(4, 240, 240)` **uint8 0–255**, and the checkpoint's first BatchNorm statistics (`running_var` median 0.0216) rule out raw 0–255 inputs. ÷255 vs z-score is still undecided; channel order and label mapping are unknown. Details and the decisive test (`scripts/probe_preprocessing.py`): `verification/NPY_SLICE_INSPECTION.md`.
* **Collection facts (measured on your 1,280 slices):** every channel of every slice spans exactly 0-255, i.e. **per-slice, per-channel rescaling** (see `verification/NPY_COLLECTION_INSPECTION.md`). Channel order, orientation and masks are determined by `scripts/match_npy_to_nifti.py` (Step 1 in `docs/NEXT_STEPS_WINDOWS.md`), not assumed.
* **Open question from your folder listing:** `data/images/*.npy` (~226 KB each) hint at how your training slices were stored. To fix preprocessing properly please send: (1) your training notebook/script, (2) two `.npy` files from `data/images` and the matching files from your masks folder, (3) the code that created them. Until then `model/model_config.yaml` flags are `false`.
## Classification (unavailable)
No classifier or checkpoint exists. BraTS provides HGG/LGG grade per patient, so a patient-wise-split classifier could be trained later; it was **not** built here.
## Explainability (unavailable)
Grad-CAM/Grad-CAM++ require a classifier; TreeSHAP requires a tree model. Neither exists; nothing is displayed.
## Evaluation
`segmentation/metrics.py`: Dice, IoU, precision, recall, specificity, HD95 (mm) with explicit empty-mask policy. A trustworthy estimate needs a patient-wise held-out set unseen by the checkpoint (see `experiments/`).
