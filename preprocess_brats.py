from pathlib import Path
import re

import nibabel as nib
import numpy as np

SOURCE_DIR = Path("data/brats_subset_100/split")
MASK_DIR = Path("data/brats_subset_100/labelsTr")
OUTPUT_DIR = Path("data/brats_preprocessed_100")
IMAGE_OUT = OUTPUT_DIR / "images"
MASK_OUT = OUTPUT_DIR / "masks"

MODALITIES = ("FLAIR", "T1", "T1ce", "T2")


def normalize_mri(volume):
    """Z-score normalize nonzero voxels; keep background zero."""
    volume = volume.astype(np.float32, copy=False)
    result = np.zeros_like(volume, dtype=np.float32)

    valid = np.isfinite(volume) & (volume != 0)
    if valid.any():
        values = volume[valid]
        std = values.std()
        if std > 0:
            result[valid] = (values - values.mean()) / std

    return result


def save_nifti(data, reference, output_path, dtype):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    result = nib.Nifti1Image(
        data.astype(dtype, copy=False),
        reference.affine,
        header
    )
    nib.save(result, str(output_path))


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {SOURCE_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    MASK_OUT.mkdir(parents=True, exist_ok=True)

    case_ids = set()
    for file in SOURCE_DIR.glob("*.nii"):
        match = re.fullmatch(r"(.+)_(FLAIR|T1|T1ce|T2)", file.stem)
        if match:
            case_ids.add(match.group(1))

    if not case_ids:
        raise RuntimeError(f"No split MRI files found in {SOURCE_DIR}")

    processed = 0

    for case_id in sorted(case_ids):
        mask_path = MASK_DIR / f"{case_id}.nii"
        if not mask_path.exists():
            print(f"Skipping {case_id}: mask not found")
            continue

        mask_img = nib.load(str(mask_path))
        mask = np.asanyarray(mask_img.dataobj)

        if mask.ndim != 3:
            print(f"Skipping {case_id}: mask is not 3-D")
            continue

        case_ok = True
        for modality in MODALITIES:
            input_path = SOURCE_DIR / f"{case_id}_{modality}.nii"

            if not input_path.exists():
                print(f"Skipping {case_id}: missing {modality}")
                case_ok = False
                break

            img = nib.load(str(input_path))
            volume = img.get_fdata(dtype=np.float32)

            if volume.ndim != 3 or volume.shape != mask.shape:
                print(f"Skipping {case_id}: {modality} shape mismatch")
                case_ok = False
                break

            normalized = normalize_mri(volume)
            save_nifti(
                normalized,
                img,
                IMAGE_OUT / f"{case_id}_{modality}.nii",
                np.float32
            )

        if not case_ok:
            continue

        # Keep mask class labels unchanged; do not normalize masks.
        save_nifti(mask, mask_img, MASK_OUT / f"{case_id}.nii", mask.dtype)
        processed += 1
        print(f"Processed {case_id}")

    print(f"\nFinished: {processed} cases processed.")
    print(f"Images: {IMAGE_OUT.resolve()}")
    print(f"Masks:  {MASK_OUT.resolve()}")


if __name__ == "__main__":
    main()
