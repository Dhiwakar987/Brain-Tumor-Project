import gc
import json
import random
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ZIP_PATH = Path(r"C:\Users\numes\Downloads\archive.zip")
OUTPUT_PATH = Path("brats_train_val_split.json")
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

def normalize_path(path):
    path = path.replace("\\", "/").removeprefix("./").lower()
    if path.endswith(".nii.gz"):
        path = path[:-3]
    return path

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH) as z:
    names = z.namelist()
    normalized_names = {normalize_path(n): n for n in names}

    dataset_json_name = next(
        (n for n in names if n.lower().endswith("dataset.json")),
        None
    )
    if not dataset_json_name:
        raise FileNotFoundError("dataset.json was not found inside the ZIP.")

    metadata = json.loads(z.read(dataset_json_name).decode("utf-8-sig"))
    cases = metadata.get("training", [])
    print("Labeled training cases listed:", len(cases))

    all_values = set()
    label_errors = []

    # Ignore Windows cleanup errors if a library temporarily keeps a file open.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        for i, case in enumerate(cases, start=1):
            label_path = case.get("label")
            if not label_path:
                label_errors.append("Training entry has no label path.")
                continue

            member = normalized_names.get(normalize_path(label_path))
            if not member:
                label_errors.append(f"Label not found in ZIP: {label_path}")
                continue

            try:
                extracted = Path(z.extract(member, temp_dir))
                label_img = nib.load(str(extracted), mmap=False)
                label_data = np.asanyarray(label_img.dataobj)
                all_values.update(int(v) for v in np.unique(label_data))

                del label_data, label_img
                gc.collect()

            except Exception as error:
                label_errors.append(f"{label_path}: {error}")

            if i % 50 == 0 or i == len(cases):
                print(f"Checked labels: {i}/{len(cases)}")

    print("\n=== LABEL CHECK ===")
    print("Distinct label values found:", sorted(all_values))
    print("Label files with errors:", len(label_errors))

    for error in label_errors[:10]:
        print(error)

    if label_errors:
        print("\nSplit not created. Fix the label errors first.")
        raise SystemExit(1)

    shuffled = list(cases)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    val_count = round(len(shuffled) * VALIDATION_FRACTION)
    validation = shuffled[:val_count]
    train = shuffled[val_count:]

    split = {
        "seed": RANDOM_SEED,
        "train_count": len(train),
        "validation_count": len(validation),
        "train": train,
        "validation": validation
    }

    OUTPUT_PATH.write_text(json.dumps(split, indent=2), encoding="utf-8")

print("\n=== SPLIT CREATED ===")
print("Training cases:", len(train))
print("Validation cases:", len(validation))
print("Saved split file:", OUTPUT_PATH.resolve())
print("The separate test set was not included.")
