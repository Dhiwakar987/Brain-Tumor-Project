from pathlib import Path
import csv
import nibabel as nib

root = Path(r"C:\BraTS_Data")
images_dir = root / "imagesTr"
labels_dir = root / "labelsTr"

def index_files(folder):
    files = {}
    for p in folder.iterdir():
        if p.name.endswith(".nii.gz"):
            files[p.name[:-7]] = p
        elif p.name.endswith(".nii"):
            files[p.name[:-4]] = p
    return files

images = index_files(images_dir)
labels = index_files(labels_dir)
paired = sorted(set(images) & set(labels))
missing_images = sorted(set(labels) - set(images))
missing_labels = sorted(set(images) - set(labels))

rows = []
problems = []

for i, case in enumerate(paired, 1):
    try:
        image_shape = tuple(nib.load(str(images[case])).shape)
        label_shape = tuple(nib.load(str(labels[case])).shape)

        spatial_match = image_shape[:3] == label_shape[:3]
        four_modalities = len(image_shape) == 4 and image_shape[3] == 4
        label_is_3d = len(label_shape) == 3

        status = "OK" if spatial_match and four_modalities and label_is_3d else "CHECK"
        if status != "OK":
            problems.append(case)

        rows.append([
            case, str(image_shape), str(label_shape),
            spatial_match, four_modalities, label_is_3d, status
        ])
    except Exception as e:
        problems.append(case)
        rows.append([case, "", "", False, False, False, f"ERROR: {e}"])

    if i % 25 == 0 or i == len(paired):
        print(f"Checked {i}/{len(paired)} cases")

out = Path(r"C:\Users\numes\OneDrive\Documents\Major_Project\brats_image_label_dimension_check.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case", "image_shape", "label_shape",
        "spatial_match", "image_has_4_modalities", "label_is_3d", "status"
    ])
    writer.writerows(rows)

print("\n=== SUMMARY ===")
print("Images:", len(images))
print("Labels:", len(labels))
print("Paired cases checked:", len(paired))
print("Cases needing review:", len(problems))
print("Images missing labels:", len(missing_labels))
print("Labels missing images:", len(missing_images))
print("CSV:", out)
if problems:
    print("Review examples:", problems[:15])
