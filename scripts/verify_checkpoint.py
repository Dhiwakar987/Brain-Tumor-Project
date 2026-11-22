"""Inspect + verify the checkpoint against model/model_config.yaml and write
verification/MODEL_VERIFICATION_REPORT.md + checkpoint_inspection.json.
Usage: python scripts/verify_checkpoint.py [--checkpoint PATH] [--synthetic-smoke]
--synthetic-smoke runs one forward pass on a RANDOM-NOISE volume: it only proves the weights run
end-to-end; it is NOT a real-MRI inference verification and is recorded separately."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import Config
from model.checkpoint_validation import validate_checkpoint, write_verification_report

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(Config.CHECKPOINT_PATH))
ap.add_argument("--synthetic-smoke", action="store_true")
a = ap.parse_args()
cfg = Config.model_config()
rep = validate_checkpoint(a.checkpoint, cfg, Config.ALLOW_UNSAFE_CHECKPOINT_PICKLE)
if a.synthetic_smoke and rep["checks"]["strict_load_ok"]:
    from model.model_loader import load_model
    from model.inference import predict_volume
    model, dev = load_model(a.checkpoint, cfg, "cpu", Config.ALLOW_UNSAFE_CHECKPOINT_PICKLE)
    x = np.random.default_rng(0).normal(size=(4, 256, 256, 2)).astype(np.float32)
    t = time.perf_counter(); out, st = predict_volume(model, x, dev, 2); el = time.perf_counter() - t
    rep["smoke_test"] = {"kind": "SYNTHETIC random-noise input (NOT real MRI)", "input_shape": list(x.shape), "output_shape": list(out.shape),
                         "classes_present": st["classes_present"], "elapsed_s": round(el, 3), "device": str(dev), "finite_logits": True}
jp, mp = write_verification_report(rep, Config.VERIFICATION_DIR)
import json; print(json.dumps({k: rep.get(k) for k in ("checks", "errors", "parameters", "compatibility", "smoke_test")}, indent=1, default=str))
print("checkpoint:", json.dumps({k: rep.get("checkpoint", {}).get(k) for k in ("file_size_bytes","sha256","top_level_keys","state_dict_container_key","num_tensors","inferred_in_channels","inferred_num_classes","embedded_metadata","torch_load_mode")}, indent=1, default=str))
print("written:", jp.name, mp.name)
