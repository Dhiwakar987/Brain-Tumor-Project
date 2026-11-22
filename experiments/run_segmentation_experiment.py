"""Patient-level segmentation (+ optional paired radiomics) experiment on a LOCAL BraTS 2020 copy.
Usage:
  python experiments/run_segmentation_experiment.py --dataset D --split-csv results/split.csv --split test \
        --checkpoint model/weights/best_unet_brats.pth [--independent-confirmed] [--radiomics] [--limit N] --out results/exp1
Outputs (raw, machine-readable): per_patient_metrics.csv, summary.json, failures.csv, run_metadata.json, [radiomics_paired.csv].
Every aggregate reports n, mean, SD, median, IQR and the number of undefined (N/A) cases per metric.
Without --independent-confirmed the results are labelled 'NOT VERIFIED AS INDEPENDENT TEST' (provenance unknown)."""
import argparse, csv, json, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import Config
from model.checkpoint_validation import sha256_file
from segmentation.metrics import EMPTY_MASK_POLICY, evaluate_regions
from segmentation.nifti_io import load_nifti
from segmentation.prediction import run_segmentation
from segmentation.regions import generate_region_masks, validate_label_values
from model.preprocessing import validate_modalities

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=Config.BRATS_DATASET_PATH); ap.add_argument("--split-csv", default=None, help="optional patient-wise split CSV; without it ALL patient folders are used and results are labelled NOT independent")
ap.add_argument("--split", default="test"); ap.add_argument("--checkpoint", default=str(Config.CHECKPOINT_PATH))
ap.add_argument("--out", default="results/experiment"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--independent-confirmed", action="store_true"); ap.add_argument("--radiomics", action="store_true")
ap.add_argument("--device", default=Config.DEVICE)
a = ap.parse_args()
if not a.dataset or not Path(a.dataset).exists(): sys.exit("BraTS dataset path missing: set BRATS_DATASET_PATH or --dataset")
if a.split_csv:
    split = pd.read_csv(a.split_csv, comment="#"); ids = split[split.split == a.split].patient_id.tolist()
else:  # quick check mode: every patient folder; NOT an independent test
    ids = sorted(p.name for p in Path(a.dataset).iterdir() if p.is_dir())
if a.limit: ids = ids[:a.limit]
if not ids: sys.exit(f"No patients with split == {a.split}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = Config.model_config(); rows, fails, rad_rows = [], [], []
for pid in ids:
    d = Path(a.dataset) / pid
    try:
        vols = {m: load_nifti(d / f"{pid}_{m}.nii.gz") for m in ("t1", "t1ce", "t2", "flair")}
        gt = load_nifti(d / f"{pid}_seg.nii.gz", dtype=np.int16)
        v = validate_modalities(vols, mask=gt)
        bad = validate_label_values(gt.data)
        if not v["ok"] or bad: raise ValueError(f"validation: {v['errors']} labels:{bad}")
        t0 = time.perf_counter(); pred_path = out / "predictions" / f"{pid}_pred.nii.gz"
        res = run_segmentation(vols, d / f"{pid}_t1.nii.gz", pred_path, cfg, a.checkpoint, a.device)
        total = time.perf_counter() - t0
        pred = load_nifti(pred_path, dtype=np.int16)
        t1 = time.perf_counter(); m = evaluate_regions(pred.data, gt.data, vols["t1"].zooms); eval_s = time.perf_counter() - t1
        for region, mm in m.items():
            rows.append({"patient_id": pid, "region": region, **mm, "inference_s": res["timings"]["inference_s"], "preprocessing_s": res["timings"]["preprocessing_s"],
                         "segmentation_total_s": round(total, 3), "evaluation_s": round(eval_s, 3), "device": res["device"]})
        if a.radiomics:
            from radiomics_analysis import extractor as rx
            from radiomics_analysis.feature_export import compare_radiomics_features
            s = Config.radiomics_settings()
            p = rx.extract_all_radiomics_features(vols, generate_region_masks(pred.data), s, "predicted")
            r = rx.extract_all_radiomics_features(vols, generate_region_masks(gt.data), s, "ground_truth")
            if p["rows"] and r["rows"]:
                c = compare_radiomics_features(p["rows"], r["rows"]); c.insert(0, "patient_id", pid); rad_rows.append(c)
            fails += [{"patient_id": pid, "stage": "radiomics", **f} for f in p["failures"] + r["failures"]]
        print("done", pid)
    except Exception as exc:
        fails.append({"patient_id": pid, "stage": "pipeline", "error": f"{type(exc).__name__}: {exc}"}); print("FAILED", pid, exc)
df = pd.DataFrame(rows); df.to_csv(out / "per_patient_metrics.csv", index=False)
pd.DataFrame(fails).to_csv(out / "failures.csv", index=False)
if rad_rows: pd.concat(rad_rows).to_csv(out / "radiomics_paired.csv", index=False)
summary = {}
for region, g in df.groupby("region") if len(df) else []:
    summary[region] = {}
    for k in ("dice", "iou", "precision", "recall", "specificity", "hd95_mm"):
        x = g[k].dropna(); summary[region][k] = {"n_defined": int(len(x)), "n_undefined": int(g[k].isna().sum()),
            "mean": float(x.mean()) if len(x) else None, "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "median": float(x.median()) if len(x) else None,
            "q1": float(x.quantile(.25)) if len(x) else None, "q3": float(x.quantile(.75)) if len(x) else None}
meta = {"cohort": (f"BraTS (local copy) split='{a.split}' from {Path(a.split_csv).name}" if a.split_csv else "BraTS (local copy) ALL patient folders (quick check, no split file)"), "n_requested": len(ids), "n_processed": int(df.patient_id.nunique()) if len(df) else 0, "n_failed": len({f['patient_id'] for f in fails if f['stage']=='pipeline'}),
        "independence": "ASSERTED BY USER (--independent-confirmed)" if a.independent_confirmed else "NOT VERIFIED AS INDEPENDENT TEST (checkpoint training patients unknown)",
        "checkpoint_sha256": sha256_file(a.checkpoint), "model_config": cfg, "metric_policy": EMPTY_MASK_POLICY, "radiomics_settings": Config.radiomics_settings() if a.radiomics else None,
        "environment": {"python": platform.python_version(), "os": platform.platform(), "machine": platform.machine()}, "summary": summary}
(out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str)); (out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"independence": meta["independence"], "n_processed": meta["n_processed"], "summary": summary}, indent=1))
