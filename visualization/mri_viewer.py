"""Slice extraction for the interactive viewer and PDF snapshots.
Display intensity scaling (1st-99th percentile -> 0-255) is for VISUALISATION ONLY and never
feeds back into inference or radiomics."""
import numpy as np

VIEWS = {"axial": 2, "coronal": 1, "sagittal": 0}


def n_slices(shape, view):
    return int(shape[VIEWS[view]])


def extract_slice(vol3d: np.ndarray, view: str, index: int) -> np.ndarray:
    axis = VIEWS[view]
    index = int(np.clip(index, 0, vol3d.shape[axis] - 1))
    sl = np.take(vol3d, index, axis=axis)
    return sl.T  # rows = second in-plane axis (display "up"), cols = first axis


def display_scale(sl: np.ndarray, lo_hi) -> np.ndarray:
    lo, hi = lo_hi
    if hi <= lo:
        return np.zeros_like(sl, dtype=np.uint8)
    return np.clip((sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def display_window(vol3d: np.ndarray):
    nz = vol3d[vol3d != 0]
    if nz.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(nz, 1)), float(np.percentile(nz, 99)))


def inplane_aspect(zooms, view):
    """(row spacing / col spacing) for the transposed slice, used for correct plot aspect."""
    z = zooms
    return {"axial": z[1] / z[0], "coronal": z[2] / z[0], "sagittal": z[2] / z[1]}[view]


def region_display_mask(labels3d: np.ndarray, region: str) -> np.ndarray:
    """'ALL' -> class codes 1/2/3 (labels 1,2,4); WT/TC/ET -> binary 1."""
    from segmentation.regions import REGION_LABELS
    if region == "ALL":
        out = np.zeros(labels3d.shape, dtype=np.uint8)
        out[labels3d == 1] = 1; out[labels3d == 2] = 2; out[labels3d == 4] = 3
        return out
    return np.isin(labels3d, REGION_LABELS[region]).astype(np.uint8)


def best_slice_index(labels3d, view="axial") -> int:
    """Slice with the most tumour voxels (middle slice if no tumour)."""
    axis = VIEWS[view]
    other = tuple(a for a in range(3) if a != axis)
    counts = (labels3d > 0).sum(axis=other)
    return int(np.argmax(counts)) if counts.max() > 0 else labels3d.shape[axis] // 2


_OPP = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


def axis_labels(axcodes, view):
    """Return (x_label, y_label) describing anatomical direction of the displayed axes.
    Slice arrays are transposed: columns = first in-plane array axis, rows = second."""
    cols, rows = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}[view]
    f = lambda a: f"{_OPP.get(axcodes[a], '?')} \u2190 \u2192 {axcodes[a]}"
    return f(cols), f(rows)
