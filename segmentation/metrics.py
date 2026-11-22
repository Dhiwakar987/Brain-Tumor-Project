"""Segmentation metrics with an explicit edge-case policy.

POLICY (reported with every result, never silently converted to 0 or 1):
  * Each metric returns None when it is mathematically undefined:
      Dice/IoU  : both prediction and reference empty
      Precision : prediction empty (TP+FP = 0)
      Recall    : reference empty  (TP+FN = 0)
      Specificity: no negative voxels (TN+FP = 0)
      HD95      : prediction or reference empty
  * If exactly one of prediction/reference is empty, Dice = IoU = 0.0 (defined).
  * HD95 is in millimetres, uses voxel spacing, and is the 95th percentile of the pooled
    surface-to-surface distances in both directions (symmetric).
"""
import numpy as np
from scipy import ndimage as ndi

from .regions import REGION_LABELS

EMPTY_MASK_POLICY = (
    "Undefined metrics are reported as N/A (None), not 0 or 1. Both masks empty -> Dice/IoU "
    "undefined; one mask empty -> Dice=IoU=0; HD95 undefined unless both masks are non-empty; "
    "HD95 in mm using voxel spacing (pooled symmetric 95th percentile of surface distances)."
)


def _div(num, den):
    return None if den == 0 else float(num) / float(den)


def confusion_counts(pred: np.ndarray, ref: np.ndarray) -> dict:
    p, r = pred.astype(bool), ref.astype(bool)
    tp = int(np.count_nonzero(p & r)); fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r)); tn = int(p.size - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def dice_coefficient(pred, ref):
    c = confusion_counts(pred, ref)
    if c["tp"] + c["fp"] + c["fn"] == 0:
        return None
    return 2.0 * c["tp"] / (2.0 * c["tp"] + c["fp"] + c["fn"])


def iou_score(pred, ref):
    c = confusion_counts(pred, ref)
    return _div(c["tp"], c["tp"] + c["fp"] + c["fn"])


def precision_score(pred, ref):
    c = confusion_counts(pred, ref); return _div(c["tp"], c["tp"] + c["fp"])


def recall_score(pred, ref):
    c = confusion_counts(pred, ref); return _div(c["tp"], c["tp"] + c["fn"])


def specificity_score(pred, ref):
    c = confusion_counts(pred, ref); return _div(c["tn"], c["tn"] + c["fp"])


def _surface(mask):
    return mask & ~ndi.binary_erosion(mask)


def hausdorff95(pred, ref, spacing=(1.0, 1.0, 1.0)):
    """HD95 in mm, or None if either mask is empty."""
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() or not r.any():
        return None
    union_idx = np.nonzero(p | r)
    sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in union_idx)  # exact: surfaces lie inside bbox
    p, r = p[sl], r[sl]
    ps, rs = _surface(p), _surface(r)
    sp = tuple(float(s) for s in spacing[:3])
    d_to_ref = ndi.distance_transform_edt(~rs, sampling=sp)[ps]
    d_to_pred = ndi.distance_transform_edt(~ps, sampling=sp)[rs]
    return float(np.percentile(np.concatenate([d_to_ref, d_to_pred]), 95))


def compute_binary_metrics(pred, ref, spacing=(1.0, 1.0, 1.0)) -> dict:
    c = confusion_counts(pred, ref)
    pe, re_ = (c["tp"] + c["fp"]) == 0, (c["tp"] + c["fn"]) == 0
    notes = []
    if pe and re_: notes.append("both prediction and reference empty")
    elif pe: notes.append("prediction empty")
    elif re_: notes.append("reference empty")
    dice = dice_coefficient(pred, ref)
    iou = iou_score(pred, ref)
    return {"dice": dice, "iou": iou, "precision": _div(c["tp"], c["tp"] + c["fp"]),
            "recall": _div(c["tp"], c["tp"] + c["fn"]), "specificity": _div(c["tn"], c["tn"] + c["fp"]),
            "hd95_mm": hausdorff95(pred, ref, spacing), "pred_voxels": c["tp"] + c["fp"],
            "ref_voxels": c["tp"] + c["fn"], "notes": "; ".join(notes)}


def evaluate_regions(pred_labels, ref_labels, spacing) -> dict:
    """WT/TC/ET metrics from label maps already in BraTS convention (0,1,2,4)."""
    out = {}
    for region, labs in REGION_LABELS.items():
        out[region] = compute_binary_metrics(np.isin(pred_labels, labs), np.isin(ref_labels, labs), spacing)
    return out
