"""Create a PATIENT-WISE train/val/test split (never slice-level). All modalities/slices of a patient stay together.
Usage: python experiments/make_patient_split.py --dataset /path/to/MICCAI_BraTS2020_TrainingData --seed 42 --out results/split.csv
WARNING: a split created NOW is only independent of the supplied checkpoint if you KNOW those patients were not used to
train/tune it. This script cannot know that; the provenance flag is stored in the CSV header for transparency."""
import argparse, csv, random, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fractions", default="0.7,0.15,0.15"); ap.add_argument("--out", default="results/split.csv")
a = ap.parse_args()
root = Path(a.dataset)
pats = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / f"{p.name}_t1.nii.gz").exists())
if not pats: sys.exit("No BraTS patient folders (<id>/<id>_t1.nii.gz) found")
fr = [float(x) for x in a.fractions.split(",")]; assert abs(sum(fr) - 1) < 1e-6
rnd = random.Random(a.seed); rnd.shuffle(pats)
n = len(pats); ntr = round(fr[0] * n); nva = round(fr[1] * n)
split = {p: "train" for p in pats[:ntr]} | {p: "val" for p in pats[ntr:ntr + nva]} | {p: "test" for p in pats[ntr + nva:]}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    f.write(f"# patient-wise split; seed={a.seed}; fractions={a.fractions}; n_patients={n}; checkpoint_independence=UNKNOWN (must be established by the user)\n")
    w = csv.writer(f); w.writerow(["patient_id", "split"]); [w.writerow([p, split[p]]) for p in sorted(split)]
print({s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}, "->", out)
