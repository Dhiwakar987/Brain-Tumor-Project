from pathlib import Path
import nibabel as nib
import numpy as np

source = Path(r"C:\BraTS_Data\imagesTr\BRATS_005.nii")
output = Path(r"C:\BraTS_Data\split_for_web\BRATS_005")

if not source.exists():
    raise FileNotFoundError(f"Cannot find: {source}")

img = nib.load(str(source))
data = img.get_fdata(dtype=np.float32)

if data.ndim != 4 or data.shape[-1] != 4:
    raise ValueError(f"Expected a 4D image with 4 channels; got {data.shape}")

output.mkdir(parents=True, exist_ok=True)

# Assumes source channel order: FLAIR, T1, T1ce, T2
modalities = {"FLAIR": 0, "T1": 1, "T1ce": 2, "T2": 3}

for name, channel in modalities.items():
    header = img.header.copy()
    volume = data[..., channel]
    out_img = nib.Nifti1Image(volume, img.affine, header=header)
    out_img.set_data_dtype(np.float32)
    destination = output / f"{name}.nii.gz"
    nib.save(out_img, str(destination))
    print(f"Saved: {destination}  shape={volume.shape}")

print("\nDone. Upload these four files as one new examination.")
