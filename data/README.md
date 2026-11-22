# data/ (not shipped – place your own data here)
Datasets are never included in the project ZIP. See `docs/DATASET_SETUP.md`.
```
data/brats/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
                               BraTS20_Training_001_t1ce.nii.gz
                               BraTS20_Training_001_t2.nii.gz
                               BraTS20_Training_001_flair.nii.gz
                               BraTS20_Training_001_seg.nii.gz      <- ground truth (labels 0,1,2,4)
```
The web app takes files through the upload page, so this folder is only needed for the batch scripts in `experiments/` (`BRATS_DATASET_PATH=data/brats`).
