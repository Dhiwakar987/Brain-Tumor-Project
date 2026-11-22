import csv
import gc
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ZIP_PATH = Path(r"C:\Users\numes\Downloads\archive.zip")
OUTPUT_CSV = Path("brats_all_training_label_counts.csv")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    label_members = sorted(
        name for name in z.namelist()
        if "/labelsTr/" in "/" + name.replace("\\", "/")
        and name.lower().endswith((".nii", ".nii.gz"))
    )

    if not label_members:
        raise FileNotFoundError("No training labels found in the ZIP.")

    print(f"Found {len(label_members)} training label files.")

    class_case_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_voxel_totals = {0: 0, 1: 0, 2: 0, 3: 0}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_dir = Path(temp_dir)

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "case",
                    "background_voxels",
                    "edema_voxels",
                    "non_enhancing_tumor_voxels",
                    "enhancing_tumor_voxels",
                ],
            )
            writer.writeheader()

            for index, member in enumerate(label_members, start=1):
                filename = Path(member.replace("\\", "/")).name
                local_path = temp_dir / filename

                with z.open(member) as source, open(local_path, "wb") as target:
                    target.write(source.read())

                nii = nib.load(str(local_path), mmap=False)
                label = np.asarray(nii.dataobj)
                values, counts = np.unique(label, return_counts=True)
                count_map = {
                    int(value): int(count)
                    for value, count in zip(values, counts)
                }

                for class_id in class_case_counts:
                    count = count_map.get(class_id, 0)
                    class_voxel_totals[class_id] += count
                    if count > 0:
                        class_case_counts[class_id] += 1

                writer.writerow({
                    "case": filename,
                    "background_voxels": count_map.get(0, 0),
                    "edema_voxels": count_map.get(1, 0),
                    "non_enhancing_tumor_voxels": count_map.get(2, 0),
                    "enhancing_tumor_voxels": count_map.get(3, 0),
                })

                del label, nii
                gc.collect()

                if index % 25 == 0 or index == len(label_members):
                    print(f"Processed {index}/{len(label_members)} labels")

print("\n=== SUMMARY ===")
print("Cases scanned:", len(label_members))
print("Cases containing each class:", class_case_counts)
print("Total voxels per class:", class_voxel_totals)
print("CSV saved to:", OUTPUT_CSV.resolve())
