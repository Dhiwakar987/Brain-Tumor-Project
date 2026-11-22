"""Inspect NIfTI files BEFORE using them: dimensionality, shape, voxel spacing, affine, orientation, intensity validity,
3D-visualisation suitability, and (for several files) mutual spatial compatibility. Writes verification/MRI_FILE_VERIFICATION.md.
Usage:  python scripts/inspect_mri.py path/to/brain_image.nii [t1.nii.gz t1ce.nii.gz t2.nii.gz flair.nii.gz seg.nii.gz ...]
        python scripts/inspect_mri.py --folder path/to/BraTS20_Training_001"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib, numpy as np
from config import Config

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--folder"); a = ap.parse_args()
files = [Path(f) for f in a.files]
if a.folder: files += sorted(p for p in Path(a.folder).iterdir() if p.name.lower().endswith((".nii", ".nii.gz")))
if not files: sys.exit("No files given")
rows, info = [], {}
for f in files:
    r = {"file": f.name, "exists": f.is_file()}
    if not f.is_file(): rows.append(r); continue
    try:
        img = nib.load(str(f)); d = np.asanyarray(img.dataobj); r["shape"] = tuple(d.shape); r["ndim"] = d.ndim
        r["spacing_mm"] = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3]); r["orientation"] = "".join(nib.aff2axcodes(img.affine)); r["dtype"] = str(d.dtype)
        v = d[..., 0] if d.ndim == 4 else d
        r["finite"] = bool(np.isfinite(v).all()); r["min"] = float(np.nanmin(v)); r["max"] = float(np.nanmax(v)); r["nonzero_frac"] = round(float((v != 0).mean()), 4)
        u = np.unique(v) if v.size < 5e7 else np.array([])
        r["labels"] = [int(x) for x in u] if 0 < len(u) <= 10 and np.allclose(u, np.round(u)) else None
        r["kind"] = "label mask" if r["labels"] else "intensity image"
        r["suitable_3d"] = bool(d.ndim == 3 and min(d.shape) >= 8 and r["finite"] and r["nonzero_frac"] > 0)
        r["note"] = "4-D volume: needs a single 3-D volume (select a volume/modality)" if d.ndim == 4 else ("2-D/too thin for 3-D viewing" if not r["suitable_3d"] else "")
        info[f.name] = (img.affine, d.shape)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    rows.append(r)
names = [n for n in info]; comp = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        (A, sa), (B, sb) = info[names[i]], info[names[j]]; comp.append((names[i], names[j], sa[:3] == sb[:3] and bool(np.allclose(A, B, atol=1e-3))))
lines = ["# MRI file verification", "", "| File | Found | Shape | Spacing mm | Orientation | Kind | Finite | Labels | 3-D suitable | Note |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['file']} | {r['exists']} | {r.get('shape','')} | {r.get('spacing_mm','')} | {r.get('orientation','')} | {r.get('kind','')} | {r.get('finite','')} | {r.get('labels','')} | {r.get('suitable_3d','')} | {r.get('error', r.get('note',''))} |")
lines += ["", "## Pairwise spatial compatibility (same shape AND same affine)"] + [f"- {a_} vs {b_}: {'COMPATIBLE' if ok else 'NOT compatible'}" for a_, b_, ok in comp]
Config.VERIFICATION_DIR.mkdir(exist_ok=True, parents=True); out = Config.VERIFICATION_DIR / "MRI_FILE_VERIFICATION.md"; out.write_text("\n".join(lines)); print("\n".join(lines)); print("->", out)
