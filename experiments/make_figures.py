"""Create plots/tables from SAVED experiment outputs only (per_patient_metrics.csv, radiomics_paired.csv)."""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args()
r = Path(a.results); f = r / "per_patient_metrics.csv"
if not f.exists(): sys.exit("per_patient_metrics.csv not found - run the experiment first (NOT RUN)")
df = pd.read_csv(f); n = df.patient_id.nunique()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for i, k in enumerate(("dice", "hd95_mm")):
    data = [df[df.region == g][k].dropna() for g in ("WT", "TC", "ET")]
    ax[i].boxplot(data, labels=[f"WT (n={len(data[0])})", f"TC (n={len(data[1])})", f"ET (n={len(data[2])})"]); ax[i].set_title(f"{k} per patient (cohort n={n})"); ax[i].set_ylabel("mm" if k == "hd95_mm" else "Dice")
fig.tight_layout(); fig.savefig(r / "fig_metrics_boxplot.png", dpi=150)
t = df.groupby("region")[["inference_s", "preprocessing_s", "segmentation_total_s", "evaluation_s"]].agg(["mean", "std"]).round(3); t.to_csv(r / "table_runtime.csv")
p = r / "radiomics_paired.csv"
if p.exists():
    c = pd.read_csv(p); s = c.groupby(["region", "modality", "feature_class"]).rel_diff_percent.agg(["count", "median"]).round(3).reset_index(); s.to_csv(r / "table_radiomics_reldiff_median.csv", index=False)
print("figures/tables written to", r)
