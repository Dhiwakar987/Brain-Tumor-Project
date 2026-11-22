"""Checkpoint inspection + verification.  Nothing is called 'verified' unless it was
actually checked; every check is recorded separately in the report."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model", "net", "network")


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _load_raw(path, allow_unsafe=False):
    """torch.load with weights_only=True (safe). Falls back to unsafe pickle only if
    explicitly allowed, because unpickling untrusted files can execute code."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True), "weights_only=True"
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                "Checkpoint could not be loaded with weights_only=True "
                f"({type(exc).__name__}: {str(exc)[:200]}). If you trust the file, set "
                "ALLOW_UNSAFE_CHECKPOINT_PICKLE=1.") from exc
        return torch.load(str(path), map_location="cpu", weights_only=False), "weights_only=False (UNSAFE, explicitly allowed)"


def extract_state_dict(obj):
    """Return (state_dict, container_key or None)."""
    if isinstance(obj, dict):
        for k in STATE_DICT_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, None
    raise ValueError("Could not find a state_dict inside the checkpoint")


def inspect_checkpoint(path, allow_unsafe=False) -> dict:
    """Inspect file + top-level structure (no architecture needed)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.name}")
    obj, load_mode = _load_raw(path, allow_unsafe)
    info = {"file_name": path.name, "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "torch_load_mode": load_mode, "top_level_type": type(obj).__name__}
    if isinstance(obj, dict):
        info["top_level_keys"] = [str(k) for k in list(obj.keys())[:50]]
        meta = {k: (v if isinstance(v, (int, float, str, bool)) else type(v).__name__)
                for k, v in obj.items() if k not in STATE_DICT_KEYS and not torch.is_tensor(v)}
        info["embedded_metadata"] = {str(k): v for k, v in list(meta.items())[:30]}
    sd, container = extract_state_dict(obj)
    info["state_dict_container_key"] = container
    info["num_tensors"] = len(sd)
    info["has_module_prefix"] = all(k.startswith("module.") for k in sd)
    keys = list(sd.keys())
    first_conv = next((k for k in keys if k.endswith("conv1.weight")), None)
    head = next((k for k in reversed(keys) if "segmentation_head" in k and k.endswith("weight")), None)
    if first_conv: info["inferred_in_channels"] = int(sd[first_conv].shape[1]); info["first_conv_shape"] = list(sd[first_conv].shape)
    if head: info["inferred_num_classes"] = int(sd[head].shape[0]); info["head_shape"] = list(sd[head].shape)
    info["inferred_encoder_hint"] = "resnet-style (conv1/layerN keys)" if any(".layer4." in k for k in keys) else "unknown"
    return info


def build_architecture(model_cfg: dict):
    import segmentation_models_pytorch as smp
    if model_cfg.get("architecture", "Unet") != "Unet":
        raise ValueError("Only the smp Unet architecture is supported by this project")
    return smp.Unet(encoder_name=model_cfg["encoder_name"], encoder_weights=model_cfg.get("encoder_weights"),
                    in_channels=int(model_cfg["in_channels"]), classes=int(model_cfg["classes"]),
                    activation=model_cfg.get("activation"))


def validate_checkpoint(path, model_cfg: dict, allow_unsafe=False) -> dict:
    """Full compatibility check against the configured architecture. Returns a report dict
    with separate flags; `strict_load_ok` is only True if every key/shape matches."""
    import segmentation_models_pytorch as smp
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": {}, "errors": []}
    report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "smp": getattr(smp, "__version__", "unknown"), "os": platform.platform(),
                             "cuda_available": torch.cuda.is_available()}
    checks = report["checks"]
    checks.update(file_inspected=False, architecture_instantiated=False, strict_load_ok=False,
                  preprocessing_verified_against_training=bool(model_cfg.get("verification", {}).get("preprocessing_verified_against_training")),
                  class_mapping_verified=bool(model_cfg.get("verification", {}).get("class_mapping_verified")),
                  real_inference_completed=False, output_geometry_verified=False)
    try:
        report["checkpoint"] = inspect_checkpoint(path, allow_unsafe)
        checks["file_inspected"] = True
        obj, _ = _load_raw(path, allow_unsafe)
        sd, _ = extract_state_dict(obj)
        if all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
            report["transformations"] = ["stripped DataParallel 'module.' prefix"]
        model = build_architecture(model_cfg)
        checks["architecture_instantiated"] = True
        report["architecture"] = {"name": "smp.Unet", **{k: model_cfg.get(k) for k in
                                  ("encoder_name", "encoder_weights", "in_channels", "classes", "activation")}}
        report["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        ms = model.state_dict()
        missing = sorted(set(ms) - set(sd)); unexpected = sorted(set(sd) - set(ms))
        mism = [{"key": k, "checkpoint": list(sd[k].shape), "model": list(ms[k].shape)}
                for k in set(ms) & set(sd) if tuple(sd[k].shape) != tuple(ms[k].shape)]
        report["compatibility"] = {"missing_keys": missing[:50], "n_missing": len(missing),
                                   "unexpected_keys": unexpected[:50], "n_unexpected": len(unexpected),
                                   "shape_mismatches": mism[:50], "n_shape_mismatches": len(mism)}
        checks["strict_load_ok"] = not (missing or unexpected or mism)
        if checks["strict_load_ok"]:
            model.load_state_dict(sd, strict=True)
        report["expected_io"] = {"input": [1, int(model_cfg["in_channels"]), 256, 256],
                                 "output": [1, int(model_cfg["classes"]), 256, 256]}
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def write_verification_report(report: dict, out_dir) -> tuple:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "checkpoint_inspection.json"
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    c = report.get("checks", {})
    sym = lambda b: "YES" if b else "NO / NOT VERIFIED"
    lines = ["# Model Verification Report", "", f"Generated (UTC): {report.get('created_utc')}", "",
             "## Status of each verification step", "",
             "| Step | Status |", "|---|---|"]
    for k, v in c.items():
        lines.append(f"| {k.replace('_', ' ')} | {sym(v)} |")
    ck = report.get("checkpoint", {})
    lines += ["", "## Checkpoint", f"- File: `{ck.get('file_name')}`", f"- Size: {ck.get('file_size_bytes')} bytes",
              f"- SHA-256: `{ck.get('sha256')}`", f"- Container key: `{ck.get('state_dict_container_key')}`",
              f"- Tensors: {ck.get('num_tensors')}", f"- Inferred input channels: {ck.get('inferred_in_channels')}",
              f"- Inferred output classes: {ck.get('inferred_num_classes')}", f"- Top-level keys: {ck.get('top_level_keys')}",
              f"- Embedded metadata: {ck.get('embedded_metadata')}",
              "", "## Architecture", f"```\n{json.dumps(report.get('architecture'), indent=2)}\n```",
              f"- Parameters: {report.get('parameters')}",
              "", "## Compatibility", f"```\n{json.dumps(report.get('compatibility'), indent=2)}\n```",
              "", "## Environment", f"```\n{json.dumps(report.get('environment'), indent=2)}\n```"]
    if report.get("smoke_test"):
        lines += ["", "## Inference smoke test", f"```\n{json.dumps(report['smoke_test'], indent=2)}\n```"]
    if report.get("errors"):
        lines += ["", "## Errors", *[f"- {e}" for e in report["errors"]]]
    lines += ["", "## Interpretation",
              "A passing strict load only shows the weights fit the configured architecture. It does NOT show that the "
              "preprocessing or label mapping match training, nor that predictions are accurate. Those require the training "
              "code (not supplied) and evaluation on real BraTS data with an independent patient-wise test set."]
    mp = out_dir / "MODEL_VERIFICATION_REPORT.md"
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp
