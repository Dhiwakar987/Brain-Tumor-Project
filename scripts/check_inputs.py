"""Write verification/INPUT_AVAILABILITY.md|json: which inputs were actually found. Run at any time
(also invoked at app start via `python app.py` is NOT required)."""
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_DIR, Config


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def find(root, patterns, limit=3):
    return [str(p) for pat in patterns for p in list(Path(root).glob(pat))[:limit]] if root and Path(root).exists() else []


def main():
    items = {}
    ck = Config.CHECKPOINT_PATH
    items["checkpoint"] = {"found": ck.is_file(), "name": ck.name, "sha256": sha(ck) if ck.is_file() else None}
    mc = Config.model_config()
    items["model_config"] = {"found": Config.MODEL_CONFIG_PATH.is_file(), "verification_flags": mc.get("verification")}
    items["training_code"] = {"found": bool(mc.get("verification", {}).get("training_code_supplied")), "note": "Not supplied unless flag set to true in model_config.yaml"}
    items["preprocessing_source"] = {"found": False, "note": "Training preprocessing code not supplied; preprocessing is configurable and UNVERIFIED" if not mc["verification"]["preprocessing_verified_against_training"] else "flag set verified"}
    items["class_mapping"] = {"found": True, "verified": bool(mc["verification"]["class_mapping_verified"]), "mapping": mc["class_mapping"]}
    ds = Config.BRATS_DATASET_PATH
    items["dataset"] = {"found": bool(ds) and Path(ds).exists(), "configured": bool(ds)}
    items["reference_masks"] = {"found": bool(find(ds, ["**/*seg.nii*"], 1))}
    rep = {"created_utc": datetime.now(timezone.utc).isoformat(), "items": items}
    Config.VERIFICATION_DIR.mkdir(exist_ok=True, parents=True)
    (Config.VERIFICATION_DIR / "input_availability.json").write_text(json.dumps(rep, indent=2))
    lines = ["# Input availability report", f"Generated (UTC): {rep['created_utc']}", "", "| Item | Found | Detail |", "|---|---|---|"]
    for k, v in items.items():
        lines.append(f"| {k} | {'YES' if v.get('found') else 'NO'} | {json.dumps({a: b for a, b in v.items() if a != 'found'})[:160]} |")
    (Config.VERIFICATION_DIR / "INPUT_AVAILABILITY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
