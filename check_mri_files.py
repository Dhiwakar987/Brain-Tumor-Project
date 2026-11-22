from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
import tempfile

import nibabel as nib

zip_path = Path(r"C:\Users\numes\Downloads\archive.zip")

if not zip_path.exists():
    print(f"ZIP not found: {zip_path}")
    raise SystemExit

with ZipFile(zip_path, "r") as z:
    names = z.namelist()
    files = [n for n in names if not n.endswith("/")]

    print(f"Total files: {len(files):,}")
    print(f"Uncompressed size: {sum(z.getinfo(n).file_size for n in files) / 1024**3:.2f} GB")

    print("\n--- File types ---")
    for ext, count in Counter(Path(n).suffix.lower() for n in files).most_common():
        print(f"{ext or '[no extension]'}: {count:,}")

    if "dataset.json" not in names:
        print("\ndataset.json not found.")
        raise SystemExit

    metadata = json.loads(z.read("dataset.json").decode("utf-8"))

    print("\n--- Dataset metadata ---")
    for key in ["name", "description", "tensorImageSize", "channel_names", "modality", "labels"]:
        if key in metadata:
            print(f"\n{key}:")
            print(json.dumps(metadata[key], indent=2))

    training = metadata.get("training", [])
    testing = metadata.get("test", [])

    print(f"\nTraining entries: {len(training):,}")
    print(f"Test entries: {len(testing):,}")

    if not training:
        print("No training entries found.")
        raise SystemExit

    sample = training[0]
    image_name = sample.get("image", "").removeprefix("./")
    label_name = sample.get("label", "").removeprefix("./")

    print("\n--- Sample image and label ---")
    print("Image:", image_name)
    print("Label:", label_name)
    print("Image exists:", image_name in names)
    print("Label exists:", label_name in names)

    with tempfile.TemporaryDirectory() as temp:
        if image_name in names:
            image_path = Path(z.extract(image_name, temp))
            img = nib.load(str(image_path))
            print("\n--- Sample image details ---")
            print("Shape:", img.shape)
            print("Voxel spacing:", img.header.get_zooms())
            print("Dimensions:", len(img.shape))

        if label_name in names:
            label_path = Path(z.extract(label_name, temp))
            label_img = nib.load(str(label_path))
            print("\n--- Sample label details ---")
            print("Shape:", label_img.shape)
            print("Voxel spacing:", label_img.header.get_zooms())

    print("\n--- Folder counts ---")
    for folder in ["imagesTr/", "labelsTr/", "imagesTs/"]:
        count = sum(n.startswith(folder) and not n.endswith("/") for n in names)
        print(f"{folder}: {count:,}")

print("\nCheck complete.")
