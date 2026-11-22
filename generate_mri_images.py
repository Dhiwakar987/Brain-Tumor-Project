
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the project directory
project_path = os.path.dirname(os.path.abspath(__file__))

# Locate the MRI dataset
image_file = os.path.join(
    project_path,
    "data",
    "brats2d",
    "brats2d",
    "images",
    "BraTS20_Training_001_slice114.npy"
)

# Output folder
output_dir = os.path.join(project_path, "verification")
os.makedirs(output_dir, exist_ok=True)

# Load MRI data
image = np.load(image_file)

print("MRI image shape:", image.shape)

# MRI modality names
modalities = ["T1", "T1ce", "T2", "FLAIR"]

# Generate and save individual images
for i, modality in enumerate(modalities):

    plt.figure(figsize=(5, 5))

    plt.imshow(image[i], cmap="gray")

    plt.title(modality)

    plt.axis("off")

    output_file = os.path.join(
        output_dir,
        modality.lower() + "_demo.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight",
        pad_inches=0.1
    )

    plt.close()

    print("Saved:", output_file)

print("\nAll four MRI images generated successfully!")
