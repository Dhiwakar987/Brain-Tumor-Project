"""Physical tumour measurements from a binary mask + the voxel spacing in the NIfTI header.
Units: volume mm^3 (and mL), surface area mm^2, lengths mm.
Surface area / mesh volume come from a marching-cubes mesh (scikit-image) so they can differ slightly from
PyRadiomics' own shape features (which are the authoritative radiomics values on the Radiomics page)."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes, mesh_surface_area

from .regions import REGION_LABELS, calculate_region_volume_ml


def _spacing(zooms):
    z = np.asarray(zooms[:3], dtype=float)
    if not np.all(np.isfinite(z)) or np.any(z <= 0):
        raise ValueError(f"Invalid voxel spacing {zooms}")
    return z


def region_measurements(mask: np.ndarray, zooms) -> dict:
    z = _spacing(zooms)
    mask = np.asarray(mask).astype(bool)
    n = int(mask.sum())
    out = {"voxel_count": n, "voxel_volume_mm3": float(np.prod(z)), "volume_mm3": float(n * np.prod(z)),
           "volume_ml": calculate_region_volume_ml(mask, z), "spacing_mm": [float(v) for v in z],
           "bbox_extent_mm": None, "surface_area_mm2": None, "sphericity": None, "max_3d_diameter_mm": None,
           "surface_to_volume_ratio_per_mm": None, "note": ""}
    if n == 0:
        out["note"] = "empty region"
        return out
    idx = np.nonzero(mask)
    lo = [int(i.min()) for i in idx]; hi = [int(i.max()) for i in idx]
    out["bbox_extent_mm"] = [float((h - l + 1) * s) for l, h, s in zip(lo, hi, z)]  # axis0, axis1, axis2 of the image array
    sub = np.pad(mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1], 1)
    try:
        verts, faces, _, _ = marching_cubes(sub.astype(np.uint8), level=0.5, spacing=tuple(z))
        area = float(mesh_surface_area(verts, faces)); V = out["volume_mm3"]
        out["surface_area_mm2"] = area
        out["surface_to_volume_ratio_per_mm"] = area / V
        out["sphericity"] = float((np.pi ** (1 / 3)) * (6 * V) ** (2 / 3) / area)
        pts = verts[ConvexHull(verts).vertices] if len(verts) >= 4 else verts
        out["max_3d_diameter_mm"] = float(cdist(pts, pts).max())
    except Exception as exc:  # degenerate (flat/tiny) region
        out["note"] = f"surface-based measures unavailable: {type(exc).__name__}"
    return out


def all_region_measurements(labels: np.ndarray, zooms) -> dict:
    return {r: region_measurements(np.isin(labels, labs), zooms) for r, labs in REGION_LABELS.items()}
