import gc
import json
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ZIP_PATH = Path(r"C:\Users\numes\Downloads\archive.zip")

def normalize_path(path):
    path = path.replace("\\", "/").removeprefix("./").lower()
    if path.endswith(".nii.gz"):
        path = path[:-3]
    return path

with zipfile.ZipFile(ZIP_PATH) as z:
    names = z.namelist()
    members = {normalize_path(n): n for n in names}

    json_name = next(
        (n for n in names if n.lower().endswith("dataset.json")),
        None
    )
    if not json_name:
        raise FileNotFoundError("dataset.json not found in ZIP.")

    metadata = json.loads(z.read(json_name).decode("utf-8-sig"))
    case = metadata["training"][0]

    image_member = members.get(normalize_path(case["image"]))
    label_member = members.get(normalize_path(case["label"]))

    if not image_member or not label_member:
        raise FileNotFoundError("Could not locate the sample image or label in ZIP.")

    print("Sample image:", image_member)
    print("Sample label:", label_member)
    print("Metadata channel order:", metadata.get("channel_names", metadata.get("modality", "Not specified")))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        image_path = Path(z.extract(image_member, temp_dir))
        label_path = Path(z.extract(label_member, temp_dir))

        image_nii = nib.load(str(image_path), mmap=False)
        label_nii = nib.load(str(label_path), mmap=False)

        image = np.asarray(image_nii.dataobj)
        label = np.asarray(label_nii.dataobj)

        print("\n=== IMAGE CHECK ===")
        print("Image shape:", image.shape)
        print("Image is 3D with 4 channels:", image.ndim == 4 and image.shape[-1] == 4)
        print("All image values finite:", bool(np.isfinite(image).all()))
        print("Channel count:", image.shape[-1] if image.ndim == 4 else "Not 4D")

        print("\n=== LABEL CHECK ===")
        print("Label shape:", label.shape)
        print("Image and label spatial dimensions match:", tuple(image.shape[:3]) == tuple(label.shape[:3]))
        print("Label values:", np.unique(label).tolist())
        print("Label voxel counts:", {
            int(value): int(count)
            for value, count in zip(*np.unique(label, return_counts=True))
        })

        del image, label, image_nii, label_nii
        gc.collect()

print("\nSample case check complete.")
