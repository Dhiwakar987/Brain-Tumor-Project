"""Verify a local BraTS-style dataset and write a table (CSV + Markdown).  Nothing is modified or guessed.
Usage (PowerShell):  python scripts\\verify_dataset.py "C:\\data\\brats2020\\MICCAI_BraTS2020_TrainingData" [--limit 20] [--out verification]
Expected layout: <root>\\<PatientID>\\<PatientID>_{t1,t1ce,t2,flair,seg}.nii[.gz]   (case-insensitive, one folder per patient)
Per patient it checks: all 4 modalities + seg present, every file readable, same shape/affine/orientation across files, spacing > 0,
finite values, seg labels subset of {0,1,2,4}, and flags volumes that look like random noise. Files whose modality cannot be
identified from the name are reported as UNKNOWN (never guessed)."""
import argparse, csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from model.preprocessing import spatial_smoothness
from segmentation.nifti_io import NiftiError, load_nifti

ROLES = ("t1", "t1ce", "t2", "flair", "seg")
PAT = re.compile(r"_(t1ce|t1gd|t1|t2|flair|seg)\.nii(\.gz)?$", re.I)


def classify(name: str):
    m = PAT.search(name)
    if not m: return None
    r = m.group(1).lower(); return "t1ce" if r == "t1gd" else r


def verify_patient(pdir: Path) -> dict:
    row = {"patient": pdir.name, "issues": [], "unknown_files": []}
    found = {}
    for f in sorted(pdir.iterdir()):
        if not f.is_file() or not f.name.lower().endswith((".nii", ".nii.gz")): continue
        r = classify(f.name)
        if r is None: row["unknown_files"].append(f.name); continue
        if r in found: row["issues"].append(f"duplicate {r.upper()} file: {f.name}")
        found[r] = f
    missing = [r for r in ROLES if r not in found]
    if missing: row["issues"].append("missing: " + ", ".join(m.upper() for m in missing))
    ref = None
    for r in ROLES:
        row[f"{r}_file"] = found[r].name if r in found else ""
        if r not in found: row[f"{r}_ok"] = "MISSING"; continue
        try:
            v = load_nifti(found[r], dtype=np.float32)
        except (NiftiError, Exception) as e:
            row[f"{r}_ok"] = "CORRUPT"; row["issues"].append(f"{r.upper()} unreadable: {str(e)[:80]}"); continue
        ok = []
        if not np.isfinite(v.data).all(): ok.append("non-finite")
        if min(v.zooms) <= 0: ok.append("bad spacing")
        if r == "seg":
            bad = sorted(int(x) for x in np.unique(v.data) if int(x) not in (0, 1, 2, 4) or x != int(x))
            if bad: ok.append(f"labels {bad}")
            row["seg_labels"] = [int(x) for x in np.unique(v.data)][:8]; row["seg_tumor_voxels"] = int((v.data > 0).sum())
        elif spatial_smoothness(v.data) < 0.3: ok.append("looks like noise")
        row[f"{r}_ok"] = "OK" if not ok else "; ".join(ok)
        if ok: row["issues"].append(f"{r.upper()}: " + "; ".join(ok))
        if ref is None: ref = (r, v); row["shape"] = "x".join(map(str, v.shape)); row["spacing_mm"] = "x".join(f"{z:g}" for z in v.zooms); row["orientation"] = "".join(v.axcodes)
        else:
            if v.shape != ref[1].shape: row["issues"].append(f"{r.upper()} shape {v.shape} != {ref[0].upper()} {ref[1].shape}")
            elif not np.allclose(v.affine, ref[1].affine, atol=1e-3): row["issues"].append(f"{r.upper()} affine differs from {ref[0].upper()}")
    row["usable"] = "YES" if not row["issues"] else "NO"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="verification"); a = ap.parse_args(argv)
    root = Path(a.root)
    if not root.is_dir(): sys.exit(f"Folder not found: {root}")
    pdirs = sorted(p for p in root.iterdir() if p.is_dir())
    loose = [f.name for f in root.iterdir() if f.is_file() and f.name.lower().endswith((".nii", ".nii.gz"))]
    if a.limit: pdirs = pdirs[:a.limit]
    rows = [verify_patient(p) for p in pdirs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ["patient", "usable", "t1_ok", "t1ce_ok", "t2_ok", "flair_ok", "seg_ok", "shape", "spacing_mm", "orientation", "seg_labels", "seg_tumor_voxels", "issues", "unknown_files"]
    with open(out / "dataset_verification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({**r, "issues": " | ".join(r["issues"]), "unknown_files": " | ".join(r["unknown_files"])})
    n_ok = sum(r["usable"] == "YES" for r in rows)
    md = [f"# Dataset verification", f"Root: `{root.name}` · patients checked: {len(rows)} · fully usable: {n_ok} · with problems: {len(rows) - n_ok}", ""]
    if loose: md += [f"**Loose .nii files not inside patient folders (cannot be grouped by patient):** {', '.join(loose[:10])}", ""]
    md += ["| Patient | Usable | T1 | T1ce | T2 | FLAIR | SEG | Shape | Spacing | Orient. | Issues |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows: md.append(f"| {r['patient']} | {r['usable']} | " + " | ".join(r.get(f"{k}_ok", "") for k in ROLES) + f" | {r.get('shape','')} | {r.get('spacing_mm','')} | {r.get('orientation','')} | {'; '.join(r['issues'])[:140]} |")
    (out / "DATASET_VERIFICATION.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:60])); print(f"\nWritten: {out / 'dataset_verification.csv'} and DATASET_VERIFICATION.md"); return rows


if __name__ == "__main__":
    main()
