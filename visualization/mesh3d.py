"""Surface meshes for the 3-D viewer, computed from the REAL MRI volume and the REAL segmentation labels.
Brain/head surface = iso-surface of the (lightly smoothed) MRI at 15% of its 99th percentile of non-zero voxels.
Tumour surfaces = iso-surface of each label mask. Coordinates are in millimetres (voxel index x spacing)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

BRAIN_LEVEL_FRACTION = 0.15


def _mesh(vol, level, spacing, step):
    v = np.pad(vol, 1)
    verts, faces, _, _ = marching_cubes(v, level=level, spacing=tuple(spacing), step_size=step)
    verts = verts - np.asarray(spacing)  # undo the 1-voxel padding shift
    return {"x": np.round(verts[:, 0], 1).tolist(), "y": np.round(verts[:, 1], 1).tolist(), "z": np.round(verts[:, 2], 1).tolist(),
            "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist()}


def build_meshes(image3d, labels3d, zooms, brain_step=3) -> dict:
    z = np.asarray(zooms[:3], dtype=float)
    out = {"units": "mm", "brain": None, "regions": {}, "brain_level_rule": f"{BRAIN_LEVEL_FRACTION:.2f} x p99(non-zero intensity)", "notes": []}
    nz = image3d[image3d != 0]
    if nz.size:
        sm = ndi.gaussian_filter(image3d.astype(np.float32), 1.0)
        try:
            out["brain"] = _mesh(sm, BRAIN_LEVEL_FRACTION * float(np.percentile(nz, 99)), z, brain_step)
        except Exception as exc:
            out["notes"].append(f"brain surface unavailable: {type(exc).__name__}")
    for name, lab in (("NCR/NET (label 1)", 1), ("Edema (label 2)", 2), ("Enhancing (label 4)", 4)):
        m = (labels3d == lab)
        if m.sum() < 4:
            continue
        try:
            out["regions"][name] = _mesh(ndi.gaussian_filter(m.astype(np.float32), 0.6), 0.5, z, 1)
        except Exception as exc:
            out["notes"].append(f"{name}: {type(exc).__name__}")
    out["bounds_mm"] = [float(s * v) for s, v in zip(image3d.shape, z)]
    return out
