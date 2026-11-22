"""PyRadiomics wrapper.  (Package is called radiomics_analysis so it does not shadow the
real `radiomics` library.)  Never crashes the whole run: per-(region, modality) failures
are collected and returned."""
import logging
import time

import numpy as np

from segmentation.nifti_io import to_sitk

log = logging.getLogger(__name__)
FAMILIES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
REGIONS = ("WT", "TC", "ET")


def pyradiomics_version() -> str:
    import radiomics
    return getattr(radiomics, "__version__", "unknown")


def initialize_radiomics_extractor(settings: dict, families=FAMILIES):
    """Create a RadiomicsFeatureExtractor from a settings dict (same structure as settings.yaml)."""
    from radiomics import featureextractor
    s = {"imageType": settings.get("imageType", {"Original": {}}), "setting": dict(settings.get("setting", {})),
         "featureClass": {f: [] for f in families}}
    ex = featureextractor.RadiomicsFeatureExtractor(s)
    return ex


def validate_radiomics_geometry(image, mask, atol=1e-3) -> dict:
    """image / mask are NiftiVolume (data, affine, zooms). Returns {ok, errors}."""
    errors = []
    if image.shape != mask.shape: errors.append(f"shape mismatch {image.shape} vs {mask.shape}")
    if not np.allclose(image.affine, mask.affine, atol=atol): errors.append("affine mismatch")
    if not np.all(np.isfinite(image.data)): errors.append("image has non-finite values")
    return {"ok": not errors, "errors": errors}


def _run(settings, families, image_vol, mask_arr, region, modality, source):
    """Execute PyRadiomics once, return list of rows."""
    if int(np.count_nonzero(mask_arr)) == 0:
        raise ValueError("empty region")
    ex = initialize_radiomics_extractor(settings, families)
    img = to_sitk(image_vol.data.astype(np.float32), image_vol.affine, image_vol.zooms)
    msk = to_sitk(mask_arr.astype(np.uint8), image_vol.affine, image_vol.zooms)
    res = ex.execute(img, msk, label=1)
    rows = []
    for k, v in res.items():
        if k.startswith("diagnostics_"):
            continue
        _, fam, name = k.split("_", 2)
        try:
            val = float(np.asarray(v).item())
        except Exception:
            continue
        rows.append({"region": region, "modality": modality, "segmentation_source": source,
                     "feature_class": fam, "feature_name": name, "value": val})
    return rows


def _family_fn(family):
    def fn(image_vol, mask_arr, settings, region="WT", modality="t1", source="predicted"):
        return _run(settings, (family,), image_vol, mask_arr, region, modality, source)
    fn.__name__ = f"extract_{family}_features"
    fn.__doc__ = f"Extract only the {family} feature family for one image/mask pair."
    return fn


extract_shape_features = _family_fn("shape")
extract_firstorder_features = _family_fn("firstorder")
extract_glcm_features = _family_fn("glcm")
extract_glrlm_features = _family_fn("glrlm")
extract_glszm_features = _family_fn("glszm")
extract_gldm_features = _family_fn("gldm")
extract_ngtdm_features = _family_fn("ngtdm")


def extract_all_radiomics_features(volumes: dict, region_masks: dict, settings: dict, source="predicted",
                                   regions=REGIONS, modalities=None, progress_cb=None) -> dict:
    """volumes: modality -> NiftiVolume. region_masks: region -> bool array.
    Shape features are mask-only, so they are extracted once per region (modality 'mask_only')
    to avoid duplicated identical rows. Returns {rows, failures, timings, settings, pyradiomics_version}."""
    modalities = list(modalities or volumes.keys())
    rows, failures, timings = [], [], {}
    first = modalities[0]
    for region in regions:
        mask = region_masks[region]
        geom_ok = validate_radiomics_geometry(volumes[first], type("M", (), {"shape": mask.shape,
                                              "affine": volumes[first].affine, "data": mask})())
        for modality in modalities:
            t0 = time.perf_counter()
            fams = FAMILIES if modality == first else tuple(f for f in FAMILIES if f != "shape")
            try:
                if not geom_ok["ok"]:
                    raise ValueError("; ".join(geom_ok["errors"]))
                r = _run(settings, fams, volumes[modality], mask, region, modality, source)
                for row in r:
                    if row["feature_class"] == "shape":
                        row["modality"] = "mask_only"
                rows.extend(r)
            except Exception as exc:  # keep going; record failure
                msg = f"{type(exc).__name__}: {exc}"
                log.warning("Radiomics failed for %s/%s: %s", region, modality, msg)
                failures.append({"region": region, "modality": modality, "error": msg})
            timings[f"{region}/{modality}"] = round(time.perf_counter() - t0, 3)
            if progress_cb: progress_cb(region, modality)
    return {"rows": rows, "failures": failures, "timings": timings, "settings": settings,
            "pyradiomics_version": pyradiomics_version(), "segmentation_source": source}
