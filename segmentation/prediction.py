"""End-to-end segmentation pipeline: validated volumes -> BraTS-label NIfTI."""
import time

import numpy as np

from model.inference import map_prediction_labels, predict_volume
from model.model_loader import load_model
from model.preprocessing import invert_orientation, preprocess_mri, unpad_prediction
from .nifti_io import save_nifti_like, verify_saved_geometry
from .regions import calculate_region_volume_ml, generate_region_masks


def quality_checks(labels: np.ndarray, zooms, mask_volumes_ml=None) -> list:
    """Heuristic WARNINGS only (never modify the prediction). These are NOT a validated quality score."""
    warns = []
    masks = generate_region_masks(labels)
    wt = int(masks["WT"].sum())
    if wt == 0:
        warns.append("Predicted mask is empty (no tumour voxels found).")
    else:
        vol = calculate_region_volume_ml(masks["WT"], zooms)
        if vol < 0.5:
            warns.append(f"Very small predicted whole-tumour volume ({vol:.2f} mL); possible false positive or missed tumour.")
        if vol > 600:
            warns.append(f"Implausibly large predicted whole-tumour volume ({vol:.0f} mL); check preprocessing/orientation.")
        # fragmentation heuristic
        from scipy import ndimage as ndi
        _, n_comp = ndi.label(masks["WT"])
        if n_comp > 20:
            warns.append(f"Prediction is fragmented ({n_comp} disconnected components).")
    if masks["ET"].sum() > 0 and masks["WT"].sum() < masks["ET"].sum():
        warns.append("Enhancing region larger than whole tumour (inconsistent label mapping?).")
    return warns


def run_segmentation(volumes: dict, reference_path, out_path, model_cfg: dict, checkpoint_path,
                     device_preference="auto", batch_size=8, allow_unsafe=False, stage_cb=None) -> dict:
    """Run preprocessing + inference + label mapping + NIfTI saving. Returns a result dict with timings."""
    stage = stage_cb or (lambda s: None)
    timings = {}
    t = time.perf_counter(); stage("loading_model")
    model, device = load_model(checkpoint_path, model_cfg, device_preference, allow_unsafe)
    timings["model_load_s"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter(); stage("preprocessing")
    stack, transform = preprocess_mri(volumes, model_cfg)
    timings["preprocessing_s"] = round(time.perf_counter() - t, 3)

    stage("inference")
    pred_cls, stats = predict_volume(model, stack, device, batch_size=batch_size,
                                     skip_empty=bool(model_cfg["preprocessing"].get("skip_empty_slices", False)))
    timings["inference_s"] = stats["elapsed_s"]

    t = time.perf_counter(); stage("postprocessing")
    pred_cls = unpad_prediction(pred_cls, transform["pad_hw"])
    pred_cls = invert_orientation(pred_cls, transform.get("slice_orientation", "identity"), axes=(0, 1))  # back to the NIfTI grid
    labels = map_prediction_labels(pred_cls, model_cfg["class_mapping"])
    ref = volumes[model_cfg["modality_order"][0]]
    assert labels.shape == ref.shape, "Prediction shape does not match input geometry"
    masks = generate_region_masks(labels)
    timings["region_masks_s"] = round(time.perf_counter() - t, 3)
    stage("saving")
    save_nifti_like(labels, reference_path, out_path, dtype=np.uint8)
    geom = verify_saved_geometry(out_path, reference_path)
    result = {
        "device": stats["device"], "inference_stats": stats, "transform": transform,
        "class_mapping": model_cfg["class_mapping"], "output_geometry_check": geom,
        "label_voxel_counts": {int(v): int((labels == v).sum()) for v in np.unique(labels)},
        "region_volumes_ml": {r: round(calculate_region_volume_ml(m, ref.zooms), 4) for r, m in masks.items()},
        "region_voxels": {r: int(m.sum()) for r, m in masks.items()},
        "qc_warnings": quality_checks(labels, ref.zooms), "timings": timings,
    }
    return result
