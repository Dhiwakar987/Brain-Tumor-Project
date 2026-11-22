from pathlib import Path
import csv
import random
import numpy as np
import nibabel as nib

DATA_ROOT = Path(r"C:\BraTS_Data")
PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
OUTPUT = PROJECT / "prepared_3d_patches"
PATCH = (64, 64, 64)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def index_files(folder):
    result = {}
    for p in folder.iterdir():
        if p.name.endswith(".nii.gz"):
            result[p.name[:-7]] = p
        elif p.name.endswith(".nii"):
            result[p.name[:-4]] = p
    return result

def start_for(center, size, full_size):
    return max(0, min(center - size // 2, full_size - size))

images = index_files(DATA_ROOT / "imagesTr")
labels = index_files(DATA_ROOT / "labelsTr")
cases = sorted(set(images) & set(labels))

if len(cases) != 484:
    raise RuntimeError(f"Expected 484 paired cases; found {len(cases)}")

random.shuffle(cases)
train_cases = set(cases[:387])
val_cases = set(cases[387:])

for split in ("train", "val"):
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

manifest_rows = []

for i, case in enumerate(sorted(cases), 1):
    split = "train" if case in train_cases else "val"

    image = np.asarray(nib.load(str(images[case])).dataobj, dtype=np.float32)
    label = np.asarray(nib.load(str(labels[case])).dataobj, dtype=np.uint8)

    if image.ndim != 4 or image.shape[3] != 4:
        raise RuntimeError(f"{case}: expected (X,Y,Z,4), got {image.shape}")
    if image.shape[:3] != label.shape:
        raise RuntimeError(f"{case}: image and label dimensions differ")
    if not set(np.unique(label)).issubset({0, 1, 2, 3}):
        raise RuntimeError(f"{case}: unexpected label values {np.unique(label)}")

    # Normalize each modality independently over nonzero voxels.
    for c in range(4):
        channel = image[..., c]
        mask = channel != 0
        if mask.any():
            mean = float(channel[mask].mean())
            std = float(channel[mask].std())
            channel[mask] = (channel[mask] - mean) / (std if std > 1e-8 else 1.0)
        channel[~mask] = 0

    # Center the patch on a tumor voxel when available.
    tumor_z = np.where(np.any(label > 0, axis=(0, 1)))[0]
    if len(tumor_z):
        z = int(random.choice(tumor_z))
        xy = np.argwhere(label[:, :, z] > 0)
        x, y = map(int, xy[random.randrange(len(xy))])
        center = (x, y, z)
    else:
        center = tuple(n // 2 for n in label.shape)

    starts = tuple(start_for(center[a], PATCH[a], label.shape[a]) for a in range(3))
    x0, y0, z0 = starts
    sx, sy, sz = PATCH

    image_patch = image[x0:x0+sx, y0:y0+sy, z0:z0+sz, :]
    label_patch = label[x0:x0+sx, y0:y0+sy, z0:z0+sz]

    # Save image as (channels, X, Y, Z); label remains (X, Y, Z).
    image_patch = np.transpose(image_patch, (3, 0, 1, 2)).astype(np.float16)
    label_patch = label_patch.astype(np.uint8)

    out_file = OUTPUT / split / f"{case}.npz"
    np.savez_compressed(out_file, image=image_patch, label=label_patch)

    manifest_rows.append([
        case, split, str(center), str(starts),
        str(image_patch.shape), str(label_patch.shape), str(out_file)
    ])

    if i % 10 == 0 or i == len(cases):
        print(f"Processed {i}/{len(cases)} cases")

with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case", "split", "center_xyz", "patch_start_xyz",
        "image_shape", "label_shape", "file"
    ])
    writer.writerows(manifest_rows)

print("\n=== 3D PATCH PREPARATION COMPLETE ===")
print("Cases:", len(cases))
print("Training cases:", len(train_cases))
print("Validation cases:", len(val_cases))
print("Patches saved:", len(manifest_rows))
print("Image shape per patch: (4, 64, 64, 64)")
print("Label shape per patch: (64, 64, 64)")
print("Output:", OUTPUT)
print("Manifest:", OUTPUT / "manifest.csv")
