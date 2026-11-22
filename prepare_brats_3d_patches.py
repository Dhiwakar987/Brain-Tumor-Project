from pathlib import Path
import csv
import random
import numpy as np
import nibabel as nib

DATA_ROOT = Path(r"C:\BraTS_Data")
PROJECT = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project")
OUTPUT = PROJECT / "prepared_3d_patches"
PATCH_SIZE = (96, 96, 96)
random.seed(42)
np.random.seed(42)

def index_files(folder):
    files = {}
    for p in folder.iterdir():
        if p.name.endswith(".nii.gz"):
            files[p.name[:-7]] = p
        elif p.name.endswith(".nii"):
            files[p.name[:-4]] = p
    return files

def get_start(center, patch_size, volume_size):
    return max(0, min(center - patch_size // 2, volume_size - patch_size))

images = index_files(DATA_ROOT / "imagesTr")
labels = index_files(DATA_ROOT / "labelsTr")
cases = sorted(set(images) & set(labels))

if len(cases) != 484:
    raise RuntimeError(f"Expected 484 paired cases; found {len(cases)}")

shuffled = cases.copy()
random.shuffle(shuffled)
train_cases = set(shuffled[:387])
val_cases = set(shuffled[387:])

for split in ("train", "val"):
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

rows = []

for case_index, case in enumerate(cases, 1):
    split = "train" if case in train_cases else "val"

    image = np.asarray(nib.load(str(images[case])).dataobj, dtype=np.float32)
    label = np.asarray(nib.load(str(labels[case])).dataobj, dtype=np.uint8)

    if image.ndim != 4 or image.shape[3] != 4:
        raise RuntimeError(f"{case}: unexpected image shape {image.shape}")
    if image.shape[:3] != label.shape:
        raise RuntimeError(f"{case}: image/label dimensions do not match")
    if any(image.shape[i] < PATCH_SIZE[i] for i in range(3)):
        raise RuntimeError(f"{case}: volume smaller than patch size")

    for modality in range(4):
        channel = image[..., modality]
        nonzero = channel != 0
        if np.any(nonzero):
            low, high = np.percentile(channel[nonzero], [1, 99])
            if high > low:
                channel[:] = np.clip((channel - low) / (high - low), 0, 1)
            else:
                channel[:] = 0
        channel[~nonzero] = 0

    tumor_slices = np.where(np.any(label > 0, axis=(0, 1)))[0]

    if len(tumor_slices):
        z = int(random.choice(tumor_slices))
        tumor_xy = np.argwhere(label[:, :, z] > 0)
        x, y = map(int, tumor_xy[random.randrange(len(tumor_xy))])
        center = (x, y, z)
    else:
        center = tuple(size // 2 for size in label.shape)

    starts = tuple(
        get_start(center[i], PATCH_SIZE[i], label.shape[i])
        for i in range(3)
    )
    x0, y0, z0 = starts
    px, py, pz = PATCH_SIZE

    image_patch = image[x0:x0+px, y0:y0+py, z0:z0+pz, :]
    label_patch = label[x0:x0+px, y0:y0+py, z0:z0+pz]

    image_patch = np.transpose(image_patch, (3, 0, 1, 2)).astype(np.float16)
    label_patch = label_patch.astype(np.uint8)

    out_path = OUTPUT / split / f"{case}.npz"
    np.savez_compressed(out_path, image=image_patch, label=label_patch)

    rows.append([
        case, split, str(center), str(starts),
        str(image_patch.shape), str(label_patch.shape),
        int(np.count_nonzero(label_patch)), str(out_path)
    ])

    if case_index % 10 == 0 or case_index == len(cases):
        print(f"Processed {case_index}/{len(cases)} cases")

manifest_path = OUTPUT / "manifest.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case", "split", "patch_center_xyz", "patch_start_xyz",
        "image_shape", "label_shape", "tumor_voxels", "file"
    ])
    writer.writerows(rows)

print("\n=== 3D PATCH PREPARATION COMPLETE ===")
print("Cases processed:", len(cases))
print("Training cases:", len(train_cases))
print("Validation cases:", len(val_cases))
print("3D patches saved:", len(rows))
print("Image patch shape: (4, 96, 96, 96)")
print("Label patch shape: (96, 96, 96)")
print("Output folder:", OUTPUT)
print("Manifest:", manifest_path)
