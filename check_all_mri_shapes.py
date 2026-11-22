import io
import zipfile
from pathlib import Path
import nibabel as nib

ZIP_PATH = Path(r"C:\Users\numes\Downloads\archive.zip")

if not ZIP_PATH.exists():
    print("ZIP not found:", ZIP_PATH)
    raise SystemExit(1)

with zipfile.ZipFile(ZIP_PATH) as z:
    names = z.namelist()

    image_files = [
        name for name in names
        if name.lower().endswith((".nii", ".nii.gz"))
        and any(
            part.lower() in ("imagestr", "imagests")
            for part in name.replace("\\", "/").split("/")
        )
    ]

    print("MRI image files found:", len(image_files))

    checked = 0
    four_channel_count = 0
    other_shapes = []
    errors = []

    for i, name in enumerate(image_files, start=1):
        try:
            with z.open(name) as file:
                header_bytes = file.read(348)

            header = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))
            shape = tuple(int(d) for d in header.get_data_shape())
            checked += 1

            if len(shape) == 4 and shape[3] == 4:
                four_channel_count += 1
            else:
                other_shapes.append((name, shape))

        except Exception as error:
            errors.append((name, str(error)))

        if i % 50 == 0 or i == len(image_files):
            print(f"Checked {i}/{len(image_files)}")

    print("\n=== RESULTS ===")
    print("Successfully checked:", checked)
    print("3D volumes with 4 channels:", four_channel_count)
    print("Other shapes:", len(other_shapes))
    print("Read errors:", len(errors))

    if other_shapes:
        print("\nExamples of other shapes:")
        for name, shape in other_shapes[:20]:
            print(name, "->", shape)

    if errors:
        print("\nExamples of read errors:")
        for name, error in errors[:10]:
            print(name, "->", error)
