from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "brats_subset_100" / "split"
LABELS = ROOT / "data" / "brats_subset_100" / "labelsTr"
OUT = ROOT / "data" / "brats2d_preprocessed_100"

MODALITIES = ("FLAIR", "T1", "T1ce", "T2")
IMAGES_OUT = OUT / "images"
MASKS_OUT = OUT / "masks"
PREVIEWS_OUT = OUT / "previews"

def load_volume(path):
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

def normalize_volume(volume):
    """Normalize nonzero voxels per modality for consistent display/model input."""
    nonzero = volume[np.isfinite(volume) & (volume != 0)]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    # Robustly limit extreme intensity outliers before z-score normalization.
    low, high = np.percentile(nonzero, [1, 99])
    clipped = np.clip(volume, low, high)
    valid = np.isfinite(clipped) & (volume != 0)
    mean = clipped[valid].mean()
    std = clipped[valid].std()

    result = np.zeros_like(volume, dtype=np.float32)
    if std > 1e-8:
        result[valid] = (clipped[valid] - mean) / std
    return result

def display_window(slice_2d):
    """Scale a normalized slice to a useful grayscale display range."""
    valid = slice_2d[np.isfinite(slice_2d) & (slice_2d != 0)]
    if valid.size == 0:
        return np.zeros_like(slice_2d)
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        return np.zeros_like(slice_2d)
    shown = np.clip((slice_2d - low) / (high - low), 0, 1)
    shown[slice_2d == 0] = 0
    return shown

def main():
    if not SPLIT.is_dir() or not LABELS.is_dir():
        raise FileNotFoundError(f"Expected folders:\n{SPLIT}\n{LABELS}")

    cases = sorted(LABELS.glob("*.nii"))[:100]
    if len(cases) != 100:
        raise RuntimeError(f"Expected 100 label files, found {len(cases)}")

    # Do not mix new output with old output or silently overwrite prior work.
    if OUT.exists() and any(OUT.rglob("*")):
        raise RuntimeError(
            f"Output folder is not empty: {OUT}\n"
            "Move or rename it before running this script again."
        )

    for folder in (IMAGES_OUT, MASKS_OUT, PREVIEWS_OUT):
        folder.mkdir(parents=True, exist_ok=True)

    total_slices = 0

    for case_index, mask_path in enumerate(cases, start=1):
        case_id = mask_path.stem
        mask = load_volume(mask_path)

        modality_volumes = []
        for modality in MODALITIES:
            image_path = SPLIT / f"{case_id}_{modality}.nii"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing modality: {image_path}")

            volume = load_volume(image_path)
            if volume.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id} {modality}: "
                    f"{volume.shape} vs mask {mask.shape}"
                )
            modality_volumes.append(normalize_volume(volume))

        # BraTS masks commonly use labels 0, 1, 2, 4. Remap 4 to 3.
        mask = np.rint(mask).astype(np.uint8)
        mask[mask == 4] = 3
        if not np.isin(mask, [0, 1, 2, 3]).all():
            raise ValueError(f"Unexpected mask labels in {case_id}: {np.unique(mask)}")

        # Stack in fixed channel order: FLAIR, T1, T1ce, T2.
        image_4d = np.stack(modality_volumes, axis=0)  # (4, X, Y, Z)

        # Pick the slice with the largest labeled region for the visual preview.
        tumor_counts = np.count_nonzero(mask, axis=(0, 1))
        preview_z = int(np.argmax(tumor_counts)) if tumor_counts.max() else mask.shape[2] // 2

        for z in range(mask.shape[2]):
            # Transpose both image and mask the same way for consistent display.
            image_slice = image_4d[:, :, :, z].transpose(0, 2, 1).astype(np.float32)
            mask_slice = mask[:, :, z].T.astype(np.uint8)

            image_file = IMAGES_OUT / f"{case_id}_slice{z:03d}.npy"
            mask_file = MASKS_OUT / f"{case_id}_slice{z:03d}.npy"

            if image_file.exists() or mask_file.exists():
                raise FileExistsError(f"Duplicate output filename for {case_id}, slice {z}")

            np.save(image_file, image_slice)
            np.save(mask_file, mask_slice)
            total_slices += 1

        # Save a readable PNG with all four modalities and the tumor-mask overlay.
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, modality, volume in zip(axes, MODALITIES, modality_volumes):
            view = display_window(volume[:, :, preview_z].T)
            ax.imshow(view, cmap="gray", origin="lower")
            ax.set_title(modality)
            ax.axis("off")

        axes[3].imshow(
            display_window(modality_volumes[0][:, :, preview_z].T),
            cmap="gray",
            origin="lower"
        )
        overlay = np.ma.masked_where(mask[:, :, preview_z].T == 0,
                                     mask[:, :, preview_z].T)
        axes[3].imshow(overlay, cmap="autumn", alpha=0.45,
                       interpolation="nearest", origin="lower")
        axes[3].set_title("FLAIR + mask")
        axes[3].axis("off")

        fig.suptitle(f"{case_id} — slice {preview_z}")
        fig.tight_layout()
        fig.savefig(PREVIEWS_OUT / f"{case_id}_preview.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[{case_index}/100] {case_id}: {mask.shape[2]} slices")

    print("\nPreprocessing complete.")
    print(f"Cases: {len(cases)}")
    print(f"Image slices: {total_slices}")
    print(f"Mask slices: {total_slices}")
    print(f"PNG previews: {len(cases)}")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
