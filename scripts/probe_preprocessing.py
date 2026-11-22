"""Find which preprocessing / channel order / label mapping the checkpoint was trained with, using YOUR training-format slices.
It runs the real checkpoint on `.npy` slices (shape (4,H,W), e.g. BraTS20_Training_001_slice91.npy) and compares with their masks.
Whole-tumour Dice (pred>0 vs mask>0) is independent of the label mapping, so it is the cleanest probe of preprocessing + channel order.
NOTE: these slices were probably used for training, so Dice here is optimistic - this is a DIAGNOSTIC, not an accuracy estimate.
Usage (PowerShell):
  python scripts\\probe_preprocessing.py --images "data\\images" --masks "data\\masks" --limit 40
  (mask file is looked up as <masks>\\<same file name>; use --mask-template "{stem}_mask.npy" if yours differ; {stem} = image name without .npy)"""
import argparse, ast, itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from config import Config
from model.model_loader import load_model


def mask_to_classes(m):
    """Return (class-index mask, description). Never guesses silently: the interpretation used is reported."""
    m = np.asarray(m)
    if m.ndim == 3:
        if m.shape[0] == 1: m = m[0]
        elif m.shape[0] in (3, 4): return np.argmax(m, axis=0).astype(np.uint8), f"one-hot/multi-channel {m.shape} -> argmax over axis 0 (ASSUMPTION)"
        else: raise ValueError(f"Unsupported mask shape {m.shape}")
    vals = set(int(v) for v in np.unique(m))
    if vals <= {0, 1, 2, 3}: return m.astype(np.uint8), f"labels {sorted(vals)} used directly as class indices"
    if vals <= {0, 1, 2, 4}: o = m.copy(); o[o == 4] = 3; return o.astype(np.uint8), f"BraTS labels {sorted(vals)}: 4 -> class 3"
    if vals <= {0, 255}: return (m > 0).astype(np.uint8), "binary 0/255 mask (only whole-tumour comparable)"
    raise ValueError(f"Unrecognised mask values {sorted(vals)[:10]}")


CANDIDATES = {
    "div255": lambda x: x / 255.0,
    "zscore_nonzero_per_slice": lambda x: np.stack([((c - c[c > 0].mean()) / (c[c > 0].std() + 1e-8)) * (c > 0) if (c > 0).any() else c for c in x]),
    "div_slice_max": lambda x: np.stack([c / c.max() if c.max() > 0 else c for c in x]),
    "raw_0_255 (control, expected bad)": lambda x: x,
}


def dice(a, b):
    s = a.sum() + b.sum(); return None if s == 0 else 2.0 * (a & b).sum() / s


def predict(model, dev, imgs, pre, perm, mult=32):
    out = []
    x = np.stack([pre(im.astype(np.float32))[list(perm)] for im in imgs]).astype(np.float32)
    H, W = x.shape[2:]; ph, pw = (-H) % mult, (-W) % mult
    x = np.pad(x, ((0, 0), (0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    with torch.inference_mode():
        for i in range(0, len(x), 8):
            out.append(model(torch.from_numpy(x[i:i + 8]).to(dev)).argmax(1).cpu().numpy())
    p = np.concatenate(out); return p[:, ph // 2:p.shape[1] - (ph - ph // 2), pw // 2:p.shape[2] - (pw - pw // 2)].astype(np.uint8)


def score(pred, mask):
    wt = [d for d in (dice(p > 0, m > 0) for p, m in zip(pred, mask)) if d is not None]
    per = {c: [d for d in (dice(p == c, m == c) for p, m in zip(pred, mask)) if d is not None] for c in (1, 2, 3)}
    return {"wt_dice": float(np.mean(wt)) if wt else None, **{f"class{c}_dice": (float(np.mean(v)) if v else None) for c, v in per.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=40); ap.add_argument("--mask-template", default="{name}"); ap.add_argument("--checkpoint", default=str(Config.CHECKPOINT_PATH))
    ap.add_argument("--out", default=str(Config.VERIFICATION_DIR)); ap.add_argument("--device", default="cpu"); a = ap.parse_args(argv)
    cfg = Config.model_config(); model, dev = load_model(a.checkpoint, cfg, a.device)
    imgs, masks, desc, skipped = [], [], None, 0
    files = sorted(Path(a.images).glob("*.npy")); rng = np.random.default_rng(0); rng.shuffle(files)
    for f in files:
        mf = Path(a.masks) / a.mask_template.format(name=f.name, stem=f.stem)
        if not mf.is_file(): skipped += 1; continue
        im = np.load(f, allow_pickle=False)
        if im.ndim != 3 or im.shape[0] != 4: sys.exit(f"{f.name}: expected shape (4,H,W), got {im.shape}")
        mk, desc = mask_to_classes(np.load(mf, allow_pickle=False))
        if mk.shape != im.shape[1:]: sys.exit(f"{f.name}: image spatial size {im.shape[1:]} != mask size {mk.shape} - image and mask do not correspond (different crop/resize?). Fix this first.")
        if (mk > 0).sum() < 30: continue  # need visible tumour to compare
        imgs.append(im); masks.append(mk)
        if len(imgs) >= a.limit: break
    if not imgs: sys.exit(f"No usable image/mask pairs found ({skipped} images had no mask file). Check --masks and --mask-template.")
    masks = np.stack(masks); ident = (0, 1, 2, 3); res = []
    for name, pre in CANDIDATES.items():
        res.append({"preprocessing": name, "channel_order": "as stored (0,1,2,3)", **score(predict(model, dev, imgs, pre, ident), masks)})
    best = max((r for r in res if r["wt_dice"] is not None), key=lambda r: r["wt_dice"]); bpre = CANDIDATES[best["preprocessing"]]
    perm_res = []
    for perm in itertools.permutations(range(4)):
        perm_res.append({"preprocessing": best["preprocessing"], "channel_order": str(perm), **score(predict(model, dev, imgs, bpre, perm), masks)})
    perm_res.sort(key=lambda r: -(r["wt_dice"] or -1))
    pred = predict(model, dev, imgs, bpre, ast.literal_eval(perm_res[0]["channel_order"]))
    conf = np.zeros((4, 4), int)
    for p, m in zip(pred, masks): np.add.at(conf, (m.ravel(), p.ravel()), 1)
    def verdict(rows, what):
        v = [r["wt_dice"] for r in rows if r["wt_dice"] is not None]
        if len(v) < 2: return f"{what}: not enough defined results"
        gap = v[0] - v[1]
        return (f"{what}: CLEAR winner ({rows[0].get('channel_order') if what.startswith('Channel') else rows[0]['preprocessing']}), margin {gap:.3f} in WT Dice" if gap >= 0.05 and v[0] >= 0.5
                else f"{what}: NOT DECIDED (best WT Dice {v[0]:.3f}, margin to runner-up {gap:.3f}). Do not configure from this probe.")
    s1 = sorted(res, key=lambda r: -(r["wt_dice"] or -1))
    verdicts = [verdict(s1, "Preprocessing"), verdict(perm_res, "Channel order")]
    if len(imgs) < 10: verdicts.append(f"Only {len(imgs)} usable slice(s): too few for a reliable decision - use at least 20-40 slices from several patients.")
    if best["wt_dice"] is not None and best["wt_dice"] < 0.5: verdicts.append("Best WT Dice < 0.5: no candidate reproduces your masks - the mask may be misaligned with the image, or the preprocessing is different from every candidate.")
    rep = {"verdicts": verdicts, "n_slices_used": len(imgs), "mask_interpretation": desc, "stage1_preprocessing_candidates": res, "stage2_channel_orders_for_best_preprocessing": perm_res[:6],
           "confusion_rows=mask_class_cols=predicted_class": conf.tolist(),
           "warning": "Diagnostic on training-format slices (likely seen in training); not an accuracy estimate. Pick the config only if one candidate is clearly better; if several tie, the probe cannot decide."}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "preprocessing_probe.json").write_text(json.dumps(rep, indent=2))
    f4 = lambda v: "n/a" if v is None else f"{v:.3f}"
    md = ["# Preprocessing probe", f"Slices used: {len(imgs)} · mask: {desc}", "", "| Preprocessing | Channel order | WT Dice | class1 | class2 | class3 |", "|---|---|---|---|---|---|"]
    md += [f"| {r['preprocessing']} | {r['channel_order']} | {f4(r['wt_dice'])} | {f4(r['class1_dice'])} | {f4(r['class2_dice'])} | {f4(r['class3_dice'])} |" for r in res + perm_res[:6]]
    md += ["", "## Verdict", *[f"- {v}" for v in verdicts], "", f"Confusion (rows = mask class 0-3, cols = predicted class 0-3) with best settings:", "```", str(conf), "```", rep["warning"]]
    (out / "PREPROCESSING_PROBE.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md)); return rep


if __name__ == "__main__":
    main()
