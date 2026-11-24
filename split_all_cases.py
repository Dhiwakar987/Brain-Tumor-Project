from pathlib import Path
import nibabel as nib
import numpy as np

source = Path("data/brats_subset_100/imagesTr")
output = Path("data/brats_subset_100/split")
output.mkdir(parents=True, exist_ok=True)

modalities = ["FLAIR", "T1", "T1ce", "T2"]
files = sorted(source.glob("*.nii"))

if not files:
    raise SystemExit(f"No .nii files found in {source.resolve()}")

for path in files:
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)

    if data.ndim != 4 or data.shape[-1] != 4:
        print(f"Skipping {path.name}: unexpected shape {data.shape}")
        continue

    for channel, modality in enumerate(modalities):
        volume = nib.Nifti1Image(
            data[..., channel],
            image.affine,
            image.header.copy()
        )
        destination = output / f"{path.stem}_{modality}.nii"
        nib.save(volume, destination)

    print(f"Split {path.name}")

print(f"Finished. Split files are in: {output.resolve()}")
