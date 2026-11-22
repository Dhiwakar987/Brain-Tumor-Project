"""Determine - from DATA, not assumptions - how your .npy training slices were made from the original BraTS NIfTI cases:
which stored channel is which MRI modality, the in-plane orientation (transpose/flips), the slice axis and the intensity scaling.
It matches each stored slice against the same slice of the source NIfTI volumes (Pearson correlation after trying all
3 slice axes x 8 orientations x 4 modalities), then tests scaling variants numerically.  It can then write mask slices
(from the patient's *_seg NIfTI, with the SAME verified transform) so that scripts\\probe_preprocessing.py can run.
Usage (PowerShell):
  python scripts\\match_npy_to_nifti.py --nifti-root "C:\\data\\MICCAI_BraTS2020_TrainingData" --images "data\\images" --write-masks "data\\masks"
Nothing is guessed: a mapping is accepted only if correlation >= 0.98 with a margin >= 0.02 over the runner-up; otherwise the result is NOT DETERMINED
and no masks are written. Written masks keep the ORIGINAL BraTS labels (0,1,2,4): the model's class mapping is decided later by the probe."""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from segmentation.nifti_io import load_nifti

PAT = re.compile(r"_(t1ce|t1gd|t1|t2|flair|seg)\.nii(\.gz)?$", re.I)
NPY = re.compile(r"^(.+)_slice(\d+)\.npy$")
MODS = ("t1", "t1ce", "t2", "flair")
OPS = {  # name -> function on a 2-D array (applied after taking the slice)
    "identity": lambda x: x, "transpose": lambda x: x.T, "flipud": lambda x: x[::-1], "fliplr": lambda x: x[:, ::-1],
    "rot90": lambda x: np.rot90(x, 1), "rot180": lambda x: np.rot90(x, 2), "rot270": lambda x: np.rot90(x, 3), "transpose+rot180": lambda x: np.rot90(x.T, 2)}


def find_files(pdir: Path) -> dict:
    out = {}
    for f in pdir.iterdir():
        m = PAT.search(f.name)
        if m and f.is_file():
            r = m.group(1).lower(); out["t1ce" if r == "t1gd" else r] = f
    return out


def corr(a, b):
    """Pearson correlation over foreground pixels only (stored>0 or source>0), so the shared brain outline does not inflate scores."""
    a = a.astype(np.float64); b = b.astype(np.float64); m = (a > 0) | (b > 0)
    if m.sum() < 50: return 0.0
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


SCALINGS = {
    "per-slice min-max, round": lambda s, v: np.round((s - s.min()) / max(s.max() - s.min(), 1e-12) * 255),
    "per-slice min-max, floor": lambda s, v: np.floor((s - s.min()) / max(s.max() - s.min(), 1e-12) * 255),
    "per-slice x/max*255": lambda s, v: np.round(s / max(s.max(), 1e-12) * 255),
    "per-volume min-max, round": lambda s, v: np.round((s - v.min()) / max(v.max() - v.min(), 1e-12) * 255),
    "per-volume x/max*255": lambda s, v: np.round(s / max(v.max(), 1e-12) * 255),
}


def match_patient(pdir: Path, npys: list, min_brain=0.05) -> dict:
    files = find_files(pdir)
    miss = [m for m in MODS if m not in files]
    if miss: return {"patient": pdir.name, "determined": False, "reason": f"missing NIfTI: {miss}"}
    vols = {m: load_nifti(files[m]).data for m in MODS}
    shape = vols["t1"].shape
    use = []
    for f in npys:
        a = np.load(f, allow_pickle=False)
        if a.ndim == 3 and a.shape[0] == 4 and (a[0] > 0).mean() >= min_brain: use.append((int(NPY.match(f.name).group(2)), a))
    if not use: return {"patient": pdir.name, "determined": False, "reason": f"no stored slice with >= {min_brain:.0%} brain pixels"}
    H, W = use[0][1].shape[1:]
    score = {}  # (channel, modality, axis, op) -> list of corr
    for ch in range(4):
        for m in MODS:
            for ax in range(3):
                if shape[ax] <= max(i for i, _ in use): continue
                for op, fn in OPS.items():
                    cs = []
                    for idx, a in use:
                        sl = fn(np.take(vols[m], idx, axis=ax))
                        if sl.shape != (H, W): cs = None; break
                        cs.append(corr(a[ch], sl))
                    if cs: score[(ch, m, ax, op)] = float(np.mean(cs))
    res = {"patient": pdir.name, "n_slices_used": len(use), "slice_indices": [i for i, _ in use], "channels": {}, "determined": True}
    axops = set()
    for ch in range(4):
        ranked = sorted(((v, k) for k, v in score.items() if k[0] == ch), reverse=True)
        if not ranked: res["determined"] = False; res["reason"] = "no geometrically compatible candidate (stored slice size differs from NIfTI plane - cropped/resized?)"; return res
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else (0.0, None)  # strictest: best DIFFERENT (modality, axis, orientation) candidate
        ok = best[0] >= 0.98 and best[0] - runner[0] >= 0.02
        res["channels"][ch] = {"modality": best[1][1] if ok else "UNKNOWN", "slice_axis": best[1][2], "orientation": best[1][3], "corr": round(best[0], 4), "runner_up_corr": round(runner[0], 4), "accepted": ok}
        if ok: axops.add((best[1][2], best[1][3]))
        else: res["determined"] = False
    if res["determined"] and len(axops) != 1: res["determined"] = False; res["reason"] = f"channels disagree on axis/orientation: {sorted(axops)}"
    if res["determined"] and len({c["modality"] for c in res["channels"].values()}) != 4: res["determined"] = False; res["reason"] = "two channels matched the same modality"
    if not res["determined"]: return res
    ax, op = next(iter(axops)); res["slice_axis"], res["orientation"] = ax, op
    res["channel_order"] = [res["channels"][c]["modality"] for c in range(4)]
    sc = {}
    for name, fn in SCALINGS.items():
        d = []
        for idx, a in use:
            for ch in range(4):
                m = res["channels"][ch]["modality"]; s = OPS[op](np.take(vols[m], idx, axis=ax)); v = vols[m]
                d.append(np.abs(fn(s, v) - a[ch]))
        d = np.concatenate([x.ravel() for x in d]); sc[name] = {"mean_abs_diff": round(float(d.mean()), 4), "frac_within_1": round(float((d <= 1).mean()), 5), "frac_exact": round(float((d == 0).mean()), 5)}
    res["scaling_tests"] = sc
    bmin = min(v["mean_abs_diff"] for v in sc.values()); ties = [n for n, v in sc.items() if v["mean_abs_diff"] <= bmin + 1e-9]
    others = [v["mean_abs_diff"] for n, v in sc.items() if n not in ties]; bfrac = sc[ties[0]]["frac_within_1"]
    if bfrac >= 0.99 and (not others or min(others) > 2 * bmin + 1e-4):
        res["scaling"] = " == ".join(ties) + (" (variants are numerically identical on this data, e.g. because each slice minimum is 0)" if len(ties) > 1 else "")
    else:
        res["scaling"] = "NOT DETERMINED (no scaling variant reproduces the stored values within +-1 for >=99% of pixels)"
    res["_files"] = {k: str(v) for k, v in files.items()}
    return res


def write_masks(root_res, pdir: Path, npys: list, out_dir: Path) -> int:
    seg = load_nifti(find_files(pdir)["seg"], dtype=np.int16).data; n = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in npys:
        idx = int(NPY.match(f.name).group(2))
        if idx >= seg.shape[root_res["slice_axis"]]: continue
        m = OPS[root_res["orientation"]](np.take(seg, idx, axis=root_res["slice_axis"])).astype(np.uint8)
        np.save(out_dir / f.name, m); n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--nifti-root", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--write-masks"); ap.add_argument("--reference", help="patient id to determine the mapping on (default: patient with most stored slices)")
    ap.add_argument("--verify-patients", type=int, default=4, help="extra patients used to confirm the mapping is consistent"); ap.add_argument("--out", default="verification"); a = ap.parse_args(argv)
    root, imgs = Path(a.nifti_root), Path(a.images)
    by = {}
    for f in sorted(imgs.glob("*.npy")):
        m = NPY.match(f.name)
        if m: by.setdefault(m.group(1), []).append(f)
    avail = [p for p in by if (root / p).is_dir()]
    if not avail: sys.exit(f"None of the {len(by)} patient IDs in {imgs} has a folder inside {root}. Check --nifti-root (it must contain one folder per patient).")
    ref = a.reference or max(avail, key=lambda p: len(by[p])); r = match_patient(root / ref, by[ref]); results = [r]
    extra = [p for p in avail if p != ref][:a.verify_patients] if r["determined"] else []
    for p in extra: results.append(match_patient(root / p, by[p]))
    consistent = r["determined"] and all(x["determined"] and x["channel_order"] == r["channel_order"] and x["slice_axis"] == r["slice_axis"] and x["orientation"] == r["orientation"] for x in results[1:])
    n_masks = 0
    if a.write_masks and r["determined"] and consistent:
        for p in avail:
            if find_files(root / p).get("seg"): n_masks += write_masks(r, root / p, by[p], Path(a.write_masks))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in x.items() if k != "_files"} for x in results]
    summary = {"reference": ref, "patients_checked": [x["patient"] for x in results], "mapping_determined": r["determined"], "consistent_across_patients": consistent, "masks_written": n_masks,
               "patients_in_images": len(by), "patients_with_nifti_folder": len(avail), "details": clean}
    (out / "npy_nifti_match.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# NPY <-> NIfTI matching", f"Reference patient: `{ref}` · checked: {len(results)} patient(s) · stored patients: {len(by)} · with NIfTI folder: {len(avail)}", ""]
    for x in clean:
        md.append(f"## {x['patient']} - {'DETERMINED' if x['determined'] else 'NOT DETERMINED: ' + x.get('reason', 'see channel table')}")
        for ch, c in x.get("channels", {}).items(): md.append(f"- stored channel {ch}: **{c['modality']}** (axis {c['slice_axis']}, {c['orientation']}, corr {c['corr']}, runner-up {c['runner_up_corr']})")
        if x.get("scaling"): md += [f"- scaling: **{x['scaling']}**", f"- scaling tests: `{json.dumps(x['scaling_tests'])}`"]
        md.append("")
    md.append(f"Consistent across patients: **{consistent}** · masks written: **{n_masks}**" + (" (original BraTS labels 0/1/2/4 via the verified transform)" if n_masks else ""))
    (out / "NPY_NIFTI_MATCH.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md)); return summary


if __name__ == "__main__":
    main()
