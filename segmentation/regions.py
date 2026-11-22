"""BraTS tumour-region definitions (applied to labels already in BraTS convention 0/1/2/4).
WT = {1,2,4}   TC = {1,4}   ET = {4}"""
import numpy as np

BRATS_LABELS = {0: "Background", 1: "Necrotic / non-enhancing core (NCR/NET)",
                2: "Peritumoral edema (ED)", 4: "Enhancing tumour (ET)"}
REGION_LABELS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}
REGION_NAMES = {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing Tumor"}


def create_whole_tumor_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, REGION_LABELS["WT"])


def create_tumor_core_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, REGION_LABELS["TC"])


def create_enhancing_tumor_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, REGION_LABELS["ET"])


def generate_region_masks(labels: np.ndarray) -> dict:
    return {"WT": create_whole_tumor_mask(labels), "TC": create_tumor_core_mask(labels),
            "ET": create_enhancing_tumor_mask(labels)}


def get_region_voxel_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def calculate_region_volume_ml(mask: np.ndarray, zooms) -> float:
    """Volume (mL) = voxels * voxel_volume_mm3 / 1000.  Raises on invalid spacing."""
    z = np.asarray(zooms, dtype=float)
    if z.shape[0] < 3 or not np.all(np.isfinite(z)) or np.any(z[:3] <= 0):
        raise ValueError(f"Invalid voxel spacing: {zooms}")
    return float(get_region_voxel_count(mask) * float(np.prod(z[:3])) / 1000.0)


def validate_label_values(labels: np.ndarray, allowed=(0, 1, 2, 4)) -> list:
    """Return sorted unexpected label values (empty list = OK)."""
    return [int(v) for v in np.unique(labels) if int(v) not in allowed]
