from pathlib import Path
import nibabel as nib
import numpy as np

source = Path(r"C:\BraTS_Data\imagesTr\BRATS_004.nii")
out_dir = Path(r"C:\BraTS_Data\split_for_web\BRATS_004")
out_dir.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(f"Cannot find: {source}")

nii = nib.load(str(source))
data = np.asarray(nii.dataobj)

if data.ndim != 4 or data.shape[-1] != 4:
    raise ValueError(f"Expected a 4D image with 4 channels; got {data.shape}")

# Assumed source channel order: FLAIR, T1, T1ce, T2
channel_indices = {
    "T1": 1,
    "T1ce": 2,
    "T2": 3,
    "FLAIR": 0,
}

for modality, index in channel_indices.items():
    volume = data[..., index]
    header = nii.header.copy()
    header.set_data_shape(volume.shape)
    output = nib.Nifti1Image(volume, nii.affine, header)
    path = out_dir / f"{modality}.nii.gz"
    nib.save(output, str(path))
    print(f"Saved {modality}: {path} | shape={volume.shape}")

print("\nDone. Upload these four files into their matching slots.")

