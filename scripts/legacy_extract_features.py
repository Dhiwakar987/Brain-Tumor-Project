import os
import SimpleITK as sitk
from radiomics import featureextractor

def extract_tumor_radiomics(image_path, mask_path):
    print(f"Loading image: {image_path}")
    print(f"Loading mask: {mask_path}")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    try:
        print("Extracting radiomics features...")
        result = extractor.execute(image_path, mask_path)
        for key, value in result.items():
            if not key.startswith('diagnostics_'):
                print(f"{key}: {value}")
        print("\nFeature extraction completed successfully!")
        return result
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

if __name__ == "__main__":
    sample_image = "data/images/brain_image.nii.gz"
    sample_mask = "data/masks/brain_mask.nii.gz"
    if os.path.exists(sample_image) and os.path.exists(sample_mask):
        extract_tumor_radiomics(sample_image, sample_mask)
    else:
        print("Files not found.")
