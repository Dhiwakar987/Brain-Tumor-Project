"""System resource / timing benchmark on SYNTHETIC volumes through the real pipeline (real checkpoint if available).
Measures wall time per stage, peak RSS (psutil sampling) and CPU time. Results = hardware-specific; do not compare across machines.
Usage: python scripts/benchmark.py --shapes 128x128x64,240x240x155 --out verification/benchmark_results.json"""
import argparse, json, platform, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib, numpy as np, psutil
from config import Config
from segmentation.nifti_io import load_nifti
from segmentation.prediction import run_segmentation
from segmentation.metrics import evaluate_regions
from model.preprocessing import validate_modalities

ap = argparse.ArgumentParser(); ap.add_argument("--shapes", default="128x128x64"); ap.add_argument("--out", default="verification/benchmark_results.json")
ap.add_argument("--checkpoint", default=str(Config.CHECKPOINT_PATH)); ap.add_argument("--device", default="cpu"); ap.add_argument("--workdir", default=str(Path(__import__("tempfile").gettempdir()) / "neuroradiomics_bench")); a = ap.parse_args()
proc = psutil.Process(); peak = [0]; stop = threading.Event()
def sampler():
    while not stop.is_set(): peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.05)
threading.Thread(target=sampler, daemon=True).start()
cfg = Config.model_config(); res = []; wd = Path(a.workdir); wd.mkdir(exist_ok=True, parents=True)
for s in a.shapes.split(","):
    shape = tuple(int(x) for x in s.split("x")); rng = np.random.default_rng(0)
    vols = {}
    for m in ("t1", "t1ce", "t2", "flair"):
        d = np.zeros(shape, np.float32); x, y, _ = np.indices(shape); br = ((x - shape[0] / 2) ** 2 + (y - shape[1] / 2) ** 2) < (shape[0] * .4) ** 2
        d[br] = rng.normal(100, 10, br.sum()).clip(1); p = wd / f"{m}_{s}.nii.gz"; nib.save(nib.Nifti1Image(d, np.eye(4)), p); vols[m] = load_nifti(p)
    peak[0] = 0; cpu0 = time.process_time(); t0 = time.perf_counter(); t = {}
    t1 = time.perf_counter(); validate_modalities(vols); t["validation_s"] = round(time.perf_counter() - t1, 3)
    r = run_segmentation(vols, wd / f"t1_{s}.nii.gz", wd / f"pred_{s}.nii.gz", cfg, a.checkpoint, a.device); t.update(r["timings"])
    pred = load_nifti(wd / f"pred_{s}.nii.gz", dtype=np.int16); t1 = time.perf_counter(); evaluate_regions(pred.data, pred.data, vols["t1"].zooms); t["self_evaluation_s"] = round(time.perf_counter() - t1, 3)
    wall = time.perf_counter() - t0
    res.append({"input": "SYNTHETIC noise volume (not BraTS)", "shape": shape, "file_size_mb_each": round((wd / f"t1_{s}.nii.gz").stat().st_size / 1e6, 2), "stages_s": t,
                "wall_total_s": round(wall, 2), "cpu_time_s": round(time.process_time() - cpu0, 2), "peak_rss_mb": round(peak[0] / 1e6, 1), "output_mb": round((wd / f"pred_{s}.nii.gz").stat().st_size / 1e6, 3),
                "s_per_slice": r["inference_stats"]["s_per_slice"], "device": r["device"]})
stop.set()
env = {"os": platform.platform(), "cpu_count": psutil.cpu_count(), "ram_total_mb": round(psutil.virtual_memory().total / 1e6), "python": platform.python_version()}
try:
    import torch; env["torch"] = torch.__version__; env["cuda"] = torch.cuda.is_available(); env["torch_threads"] = torch.get_num_threads()
except Exception: pass
Path(a.out).parent.mkdir(exist_ok=True, parents=True); Path(a.out).write_text(json.dumps({"environment": env, "results": res}, indent=2))
print(json.dumps({"environment": env, "results": res}, indent=1))
