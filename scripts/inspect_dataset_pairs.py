import zipfile
import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt

zip_path = r"C:\Users\numes\Downloads\_output_.zip"
output_path = r"verification\dataset_pair_inspection.png"

os.makedirs("verification", exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    mask_files = [
        f for f in z.namelist()
        if f.startswith("brats2d/masks/BraTS20_Training_001_")
        and f.endswith(".npy")
    ]

    selected = []

    for mask_name in mask_files:
        mask = np.load(io.BytesIO(z.read(mask_name)))

        if np.any(mask != 0):
            image_name = mask_name.replace(
                "brats2d/masks/", "brats2d/images/"
            )

            if image_name in z.namelist():
                selected.append((mask_name, image_name, mask))

        if len(selected) == 3:
            break

    if not selected:
        raise RuntimeError("No matching non-empty pairs found.")

    fig, axes = plt.subplots(len(selected), 5, figsize=(15, 4 * len(selected)))

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (mask_name, image_name, mask) in enumerate(selected):
        image = np.load(io.BytesIO(z.read(image_name)))

        for channel in range(4):
            axes[row, channel].imshow(image[channel], cmap="gray")
            axes[row, channel].set_title(f"Image channel {channel}")

        axes[row, 4].imshow(image[0], cmap="gray")
        axes[row, 4].contour(
            
            mask > 0,
            levels=[0.5],
            colors=["red"],
            linewidths=1
        )
        axes[row, 4].set_title("MRI + tumour boundary")

        for ax in axes[row]:
            ax.axis("off")

        print(os.path.basename(mask_name))
        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        print("Mask labels:", np.unique(mask).tolist())
        print("---")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print("Saved visual inspection to:", output_path)
