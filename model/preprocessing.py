"""MRI validation + preprocessing.  All behaviour is driven by model/model_config.yaml.
The normalisation strategy is NOT invented here: it is whatever the config says, and the
config marks whether it was verified against the training pipeline."""
import numpy as np

SUPPORTED_NORMALIZATION = ("zscore_nonzero", "minmax_volume", "minmax_slice", "none")
ORIENTATIONS = ("identity", "transpose", "flipud", "fliplr", "rot90", "rot180", "rot270", "transpose+rot180")


def apply_orientation(x: np.ndarray, name: str, axes=(0, 1)) -> np.ndarray:
    """Apply the in-plane orientation used when the TRAINING slices were exported (names match scripts/match_npy_to_nifti.py;
    each acts on the two array axes given in `axes`, so it works on (H,W), (H,W,D) and (C,H,W,D) arrays)."""
    a, b = axes
    if name == "identity": return x
    if name == "transpose": return np.swapaxes(x, a, b)
    if name == "flipud": return np.flip(x, a)
    if name == "fliplr": return np.flip(x, b)
    if name == "rot90": return np.rot90(x, 1, axes=(a, b))
    if name == "rot180": return np.rot90(x, 2, axes=(a, b))
    if name == "rot270": return np.rot90(x, 3, axes=(a, b))
    if name == "transpose+rot180": return np.rot90(np.swapaxes(x, a, b), 2, axes=(a, b))
    raise ValueError(f"Unknown slice_orientation '{name}'. Options: {ORIENTATIONS}")


def invert_orientation(x: np.ndarray, name: str, axes=(0, 1)) -> np.ndarray:
    """Exact inverse of apply_orientation (used to bring predictions back to the NIfTI grid)."""
    a, b = axes
    if name == "rot90": return apply_orientation(x, "rot270", axes)
    if name == "rot270": return apply_orientation(x, "rot90", axes)
    if name == "transpose+rot180": return np.swapaxes(np.rot90(x, 2, axes=(a, b)), a, b)
    return apply_orientation(x, name, axes)  # identity, transpose, flips, rot180 are self-inverse


def spatial_smoothness(data: np.ndarray) -> float:
    """Mean lag-1 correlation between neighbouring voxels over the 3 axes. Anatomical MRI is smooth
    (typically > 0.8); i.i.d. random noise is ~0. A HEURISTIC only, used to warn - never to modify data."""
    d = np.asarray(data, dtype=np.float32); cs = []
    for ax in range(3):
        a = np.take(d, range(d.shape[ax] - 1), axis=ax).ravel(); b = np.take(d, range(1, d.shape[ax]), axis=ax).ravel()
        if a.std() > 0 and b.std() > 0: cs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cs)) if cs else 0.0


def validate_modalities(volumes: dict, required=("t1", "t1ce", "t2", "flair"), atol=1e-3,
                        mask=None) -> dict:
    """Check completeness, shape, affine, orientation, spacing and finite values.
    `volumes` maps modality -> segmentation.nifti_io.NiftiVolume.  Returns
    {ok, errors, warnings, details}. Never modifies the data."""
    errors, warnings, details = [], [], {}
    missing = [m for m in required if m not in volumes]
    if missing:
        errors.append(f"Missing modalities: {', '.join(missing)}")
    present = [m for m in required if m in volumes]
    if present:
        ref = volumes[present[0]]
        details["reference_modality"] = present[0]
        details["shape"] = list(ref.shape)
        details["voxel_spacing_mm"] = [round(z, 4) for z in ref.zooms]
        details["orientation"] = "".join(ref.axcodes)
        for m in present:
            v = volumes[m]
            if v.shape != ref.shape:
                errors.append(f"{m.upper()} shape {v.shape} differs from {present[0].upper()} {ref.shape}")
            if not np.allclose(v.affine, ref.affine, atol=atol):
                errors.append(f"{m.upper()} affine differs from {present[0].upper()} (scans not aligned)")
            if v.axcodes != ref.axcodes:
                errors.append(f"{m.upper()} orientation {''.join(v.axcodes)} differs from {''.join(ref.axcodes)}")
            if not np.all(np.isfinite(v.zooms)) or min(v.zooms) <= 0:
                errors.append(f"{m.upper()} has invalid voxel spacing {v.zooms}")
            nonfinite = int(np.size(v.data) - np.count_nonzero(np.isfinite(v.data)))
            if nonfinite:
                errors.append(f"{m.upper()} contains {nonfinite} NaN/Inf voxels")
            if not np.any(v.data != 0):
                errors.append(f"{m.upper()} is entirely zero")
        for m in present:
            sm = spatial_smoothness(volumes[m].data)
            details.setdefault("spatial_smoothness", {})[m] = round(sm, 3)
            if sm < 0.3:
                warnings.append(f"{m.upper()} has almost no spatial structure (neighbour-voxel correlation {sm:.2f}); it looks like random noise, not anatomical MRI")
        if max(ref.zooms) / max(min(ref.zooms), 1e-9) > 3:
            warnings.append(f"Strongly anisotropic voxel spacing {ref.zooms}; slice-wise 2-D inference may be unreliable")
        if any(abs(z - 1.0) > 0.05 for z in ref.zooms):
            warnings.append("Voxel spacing is not 1 mm isotropic (BraTS default); no resampling is performed")
    if mask is not None and present:
        ref = volumes[present[0]]
        if mask.shape != ref.shape:
            errors.append(f"Reference mask shape {mask.shape} differs from MRI {ref.shape}")
        elif not np.allclose(mask.affine, ref.affine, atol=atol):
            errors.append("Reference mask affine differs from MRI affine (not aligned)")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "details": details}


def normalize_mri(vol: np.ndarray, mode: str = "zscore_nonzero", clip_percentiles=None) -> tuple:
    """Normalise ONE modality volume. Returns (array float32, parameters dict)."""
    if mode not in SUPPORTED_NORMALIZATION:
        raise ValueError(f"Unknown normalization '{mode}'. Options: {SUPPORTED_NORMALIZATION}")
    v = vol.astype(np.float32, copy=True)
    params = {"mode": mode}
    if clip_percentiles and mode != "none":
        nz = v[v != 0]
        if nz.size:
            lo, hi = np.percentile(nz, clip_percentiles)
            v = np.clip(v, lo, hi); params["clip"] = [float(lo), float(hi)]
    if mode == "none":
        return v, params
    if mode == "zscore_nonzero":
        m = v != 0
        if m.any():
            mu, sd = float(v[m].mean()), float(v[m].std())
            v[m] = (v[m] - mu) / (sd if sd > 1e-8 else 1.0)
            params.update(mean=mu, std=sd)
        return v, params
    if mode == "minmax_volume":
        lo, hi = float(v.min()), float(v.max())
        v = (v - lo) / (hi - lo) if hi > lo else v * 0
        params.update(min=lo, max=hi)
        return v, params
    # minmax_slice (per axial slice, along last axis) -- only if training used it
    lo = v.min(axis=(0, 1), keepdims=True); hi = v.max(axis=(0, 1), keepdims=True)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (v - lo) / rng, params


def preprocess_mri(volumes: dict, model_cfg: dict) -> tuple:
    """Stack modalities in the configured channel order, normalise, and zero-pad H/W to a
    multiple of `pad_multiple`.  Returns (array (C,H,W,D) float32, transform_record)."""
    order = model_cfg["modality_order"]; pre = model_cfg["preprocessing"]
    chans, norm_params = [], {}
    for m in order:
        arr, p = normalize_mri(volumes[m].data, pre["normalization"], pre.get("clip_percentiles"))
        chans.append(arr); norm_params[m] = p
    stack = np.stack(chans, axis=0)
    orient = pre.get("slice_orientation", "identity")
    stack = apply_orientation(stack, orient, axes=(1, 2))   # match the orientation of the training slices (before padding)
    mult = int(pre.get("pad_multiple", 32))
    _, H, W, _ = stack.shape
    ph, pw = (-H) % mult, (-W) % mult
    pad = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
    if ph or pw:
        stack = np.pad(stack, ((0, 0), pad[0], pad[1], (0, 0)), mode="constant")
    record = {"channel_order": list(order), "normalization": norm_params, "pad_hw": [list(p) for p in pad],
              "original_shape": [int(s) for s in volumes[order[0]].shape], "padded_shape": list(stack.shape), "slice_orientation": orient,
              "config_verified_against_training": bool(model_cfg.get("verification", {}).get("preprocessing_verified_against_training", False))}
    return stack, record


def unpad_prediction(pred: np.ndarray, pad_hw) -> np.ndarray:
    (t, b), (l, r) = pad_hw
    H, W = pred.shape[0], pred.shape[1]
    return pred[t:H - b if b else H, l:W - r if r else W, ...]
