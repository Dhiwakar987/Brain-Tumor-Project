import json
import zipfile
from pathlib import Path

zip_path = Path(r"C:\Users\numes\Downloads\archive.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    matches = [name for name in z.namelist() if name.endswith("dataset.json")]

    if not matches:
        raise FileNotFoundError("Could not find dataset.json inside the ZIP.")

    name = matches[0]
    print("Found:", name)

    with z.open(name) as f:
        metadata = json.load(f)

print("\n=== LABEL DEFINITIONS ===")
print(json.dumps(metadata.get("labels", "No 'labels' field found"), indent=2))

print("\n=== MODALITIES ===")
print(json.dumps(metadata.get("modality", "No 'modality' field found"), indent=2))
