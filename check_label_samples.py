import gc
import json
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

zip_path = Path(r"C:\Users\numes\Downloads\archive.zip")
sample_count = 10

with zipfile.ZipFile(zip_path, "r") as z:
    label_files = sorted(
        name for name in z.namelist()
        if "/labelsTr/" in "/" + name.replace("\\", "/")
        and name.lower().endswith((".nii", ".nii.gz"))
    )

    if not label_files:
        raise FileNotFoundError("Could not find training labels in the ZIP.")

    selected = label_files[:sample_count]
    class_case_counts = {}
    print(f"Checking {len(selected)} of {len(label_files)} training labels\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        for index, name in enumerate(selected, start=1):
            local_path = Path(temp_dir) / Path(name).name

            with z.open(name) as source, open(local_path, "wb") as target:
                target.write(source.read())

            image = nib.load(str(local_path), mmap=False)
            labels = np.asarray(image.dataobj)
            values, counts = np.unique(labels, return_counts=True)
            count_map = {int(v): int(c) for v, c in zip(values, counts)}

            for value in count_map:
                class_case_counts[value] = class_case_counts.get(value, 0) + 1

            print(f"{index}. {Path(name).name}")
            print("   Shape:", labels.shape)
            print("   Voxel counts:", count_map)

            del labels, image
            gc.collect()

print("\n=== Number of sampled cases containing each class ===")
print(json.dumps(class_case_counts, indent=2))
