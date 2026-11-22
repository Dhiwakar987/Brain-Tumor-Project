"""Slice-wise 2-D U-Net inference over a preprocessed 4-channel volume."""
import time

import numpy as np
import torch


def predict_slice(model, slice_chw: np.ndarray, device) -> np.ndarray:
    """slice_chw: (C,H,W) float32 -> class-index map (H,W) uint8."""
    with torch.inference_mode():
        x = torch.from_numpy(slice_chw[None]).to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite logits")
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def predict_volume(model, volume_chwd: np.ndarray, device, batch_size=8, skip_empty=False,
                   progress_cb=None) -> tuple:
    """volume_chwd: (C,H,W,D). Returns (class volume (H,W,D) uint8, stats dict)."""
    C, H, W, D = volume_chwd.shape
    out = np.zeros((H, W, D), dtype=np.uint8)
    t0 = time.perf_counter()
    idxs = [k for k in range(D) if not (skip_empty and not np.any(volume_chwd[..., k]))]
    with torch.inference_mode():
        for start in range(0, len(idxs), batch_size):
            ids = idxs[start:start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(np.stack([volume_chwd[..., k] for k in ids]))).to(device)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite logits")
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for j, k in enumerate(ids):
                out[..., k] = pred[j]
            if progress_cb:
                progress_cb(min(start + batch_size, len(idxs)), len(idxs))
    elapsed = time.perf_counter() - t0
    stats = {"slices_total": D, "slices_predicted": len(idxs), "elapsed_s": round(elapsed, 3),
             "s_per_slice": round(elapsed / max(len(idxs), 1), 4), "device": str(device),
             "classes_present": [int(v) for v in np.unique(out)], "input_shape": list(volume_chwd.shape),
             "output_shape": list(out.shape)}
    return out, stats


def map_prediction_labels(pred_classes: np.ndarray, class_mapping: dict) -> np.ndarray:
    """Convert model class indices to BraTS labels using the configured mapping."""
    mapping = {int(k): int(v) for k, v in class_mapping.items()}
    unknown = [int(c) for c in np.unique(pred_classes) if int(c) not in mapping]
    if unknown:
        raise ValueError(f"Model produced class indices {unknown} that are not in the class mapping")
    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    return lut[pred_classes]


# Convenience re-exports so the model module exposes the full pipeline API.
from segmentation.nifti_io import save_nifti_like as save_prediction_nifti  # noqa: E402,F401
from segmentation.regions import generate_region_masks  # noqa: E402,F401
