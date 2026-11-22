import zipfile
from pathlib import Path
import nibabel as nib

zip_path = r"C:\Users\numes\Downloads\archive.zip"
image_name = "imagesTr/BRATS_457.nii"
label_name = "labelsTr/BRATS_457.nii"

with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()

    image_member = next((n for n in names if n.endswith(image_name)), None)
    label_member = next((n for n in names if n.endswith(label_name)), None)

    if not image_member:
        print("Image file not found.")
        raise SystemExit(1)

    temp = Path("temp_mri_check")
    temp.mkdir(exist_ok=True)

    image_path = Path(z.extract(image_member, temp))
    image = nib.load(str(image_path))

    print("Image shape:", image.shape)
    print("Image dimensions:", len(image.shape))
    print("Voxel spacing:", image.header.get_zooms())

    if len(image.shape) == 4 and 4 in image.shape:
        print("CONFIRMED: 4D volume with a dimension of 4, consistent with four modalities.")
    elif len(image.shape) == 3:
        print("This file is a 3D volume; modalities may be stored separately.")
    else:
        print("Check the shape and dataset metadata to interpret the channel layout.")

    if label_member:
        label_path = Path(z.extract(label_member, temp))
        label = nib.load(str(label_path))
        print("Label shape:", label.shape)
        print("Spatial dimensions match:", tuple(image.shape[:3]) == tuple(label.shape[:3]))
