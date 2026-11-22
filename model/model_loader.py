"""Model loading (strict) with caching and device selection."""
import logging
import threading
from pathlib import Path

import torch

from .checkpoint_validation import (_load_raw, build_architecture, extract_state_dict,
                                    inspect_checkpoint, sha256_file, validate_checkpoint)

log = logging.getLogger(__name__)
_cache = {}
_lock = threading.Lock()


class ModelLoadError(RuntimeError):
    pass


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("DEVICE=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, model_cfg: dict, device_preference="auto", allow_unsafe=False):
    """Instantiate the configured U-Net and load the checkpoint with strict=True.
    Raises ModelLoadError on ANY incompatibility (no key dropping, no random re-init)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelLoadError(f"Checkpoint file not found (expected at the configured CHECKPOINT_PATH: {path.name})")
    key = (str(path), path.stat().st_mtime_ns, device_preference)
    with _lock:
        if key in _cache:
            return _cache[key]
        try:
            obj, _ = _load_raw(path, allow_unsafe)
            sd, _ = extract_state_dict(obj)
            if all(k.startswith("module.") for k in sd):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            model = build_architecture(model_cfg)
            model.load_state_dict(sd, strict=True)
        except Exception as exc:
            raise ModelLoadError(f"Checkpoint is not compatible with the configured architecture: {exc}") from exc
        device = select_device(device_preference)
        model.to(device).eval()
        _cache.clear()
        _cache[key] = (model, device)
        log.info("Model loaded on %s", device)
        return model, device


def get_model_information(checkpoint_path, model_cfg: dict, device_preference="auto") -> dict:
    """Information safe to show in the UI (no absolute paths)."""
    path = Path(checkpoint_path)
    info = {"architecture": f"U-Net / {model_cfg['encoder_name']}", "in_channels": model_cfg["in_channels"],
            "classes": model_cfg["classes"], "checkpoint_present": path.is_file(),
            "checkpoint_file": path.name, "device": str(select_device(device_preference)) if device_preference != "cuda" or torch.cuda.is_available() else "cuda (unavailable)",
            "modality_order": model_cfg["modality_order"], "class_mapping": model_cfg["class_mapping"],
            "preprocessing": model_cfg["preprocessing"], "verification_flags": model_cfg.get("verification", {}),
            "torch_version": torch.__version__}
    if path.is_file():
        info["checkpoint_sha256"] = sha256_file(path)
    return info


def validate_checkpoint_file(checkpoint_path, model_cfg, allow_unsafe=False) -> dict:
    """Thin wrapper kept here so callers only need model_loader."""
    return validate_checkpoint(checkpoint_path, model_cfg, allow_unsafe)
