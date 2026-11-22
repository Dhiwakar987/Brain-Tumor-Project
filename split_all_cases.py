from pathlib import Path
import nibabel as nib
import numpy as np

source_dir = Path(r"C:\BraTS_Data\imagesTr")
output_dir = Path(r"C:\BraTS_Data\split_for_web")
modalities = {"FLAIR": 0, "T1": 1, "T1ce": 2, "T2": 3}

files = sorted(
    p for p in source_dir.iterdir()
    if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
)

print(f"Found {len(files)} source files.")
done = 0
skipped = 0

for i, source in enumerate(files, 1):
    case_id = source.name.removesuffix(".nii.gz").removesuffix(".nii")
    case_dir = output_dir / case_id

    try:
        img = nib.load(str(source))
        if len(img.shape) != 4 or img.shape[-1] != 4:
            print(f"SKIP {source.name}: expected 4 channels, got {img.shape}")
            skipped += 1
            continue

        case_dir.mkdir(parents=True, exist_ok=True)

        for name, channel in modalities.items():
            destination = case_dir / f"{name}.nii.gz"
            volume = np.asarray(img.dataobj[..., channel], dtype=np.float32)
            out_img = nib.Nifti1Image(volume, img.affine, header=img.header.copy())
            out_img.set_data_dtype(np.float32)
            nib.save(out_img, str(destination))

        done += 1
        print(f"{i}/{len(files)}: prepared {case_id}")

    except Exception as e:
        print(f"ERROR {source.name}: {e}")
        skipped += 1

print(f"\nFinished. Prepared {done} cases; skipped/failed {skipped}.")
print(f"Output folder: {output_dir}")
