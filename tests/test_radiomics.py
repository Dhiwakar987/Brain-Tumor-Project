import numpy as np
import nibabel as nib
import pytest
from conftest import make_labels, make_volume
from config import Config
from radiomics_analysis import extractor as rx
from radiomics_analysis.feature_export import compare_radiomics_features, save_radiomics_csv, save_radiomics_excel
from segmentation.nifti_io import load_nifti
from segmentation.regions import generate_region_masks


def vol(tmp_path, img, name):
    p = tmp_path / name; nib.save(img, p); return load_nifti(p)


@pytest.fixture()
def data(tmp_path):
    v = {m: vol(tmp_path, make_volume(seed=i), f"{m}.nii.gz") for i, m in enumerate(("t1", "flair"))}
    lab = load_nifti(_save(tmp_path, make_labels()), dtype=np.int16)
    return v, generate_region_masks(lab.data)


def _save(tmp_path, img):
    p = tmp_path / "lab.nii.gz"; nib.save(img, p); return p


def test_geometry_validation(tmp_path):
    a = vol(tmp_path, make_volume(), "a.nii.gz")
    ok = rx.validate_radiomics_geometry(a, a); assert ok["ok"]
    b = vol(tmp_path, make_volume((48, 48, 8)), "b.nii.gz")
    assert not rx.validate_radiomics_geometry(a, b)["ok"]
    c = vol(tmp_path, make_volume(zooms=(2, 2, 2)), "c.nii.gz")
    assert "affine mismatch" in rx.validate_radiomics_geometry(a, c)["errors"]


@pytest.mark.slow
def test_all_families_extracted(data):
    v, masks = data
    s = Config.radiomics_settings()
    out = rx.extract_all_radiomics_features(v, masks, s, "predicted", regions=("WT",), modalities=["t1"])
    fams = {r["feature_class"] for r in out["rows"]}
    assert fams == set(rx.FAMILIES), fams
    assert not out["failures"] and out["pyradiomics_version"] and out["segmentation_source"] == "predicted"
    assert all(r["modality"] == "mask_only" for r in out["rows"] if r["feature_class"] == "shape")
    assert all(np.isfinite(r["value"]) for r in out["rows"])


@pytest.mark.slow
def test_single_family_functions(data):
    v, masks = data
    rows = rx.extract_glcm_features(v["t1"], masks["WT"], Config.radiomics_settings(), "WT", "t1")
    assert rows and {r["feature_class"] for r in rows} == {"glcm"}
    rows = rx.extract_ngtdm_features(v["t1"], masks["TC"], Config.radiomics_settings(), "TC", "t1")
    assert {r["feature_class"] for r in rows} == {"ngtdm"}


def test_empty_region_recorded_not_crash(tmp_path):
    v = {"t1": vol(tmp_path, make_volume(), "t1.nii.gz")}
    masks = generate_region_masks(load_nifti(_save(tmp_path, make_labels(empty=True)), dtype=np.int16).data)
    out = rx.extract_all_radiomics_features(v, masks, Config.radiomics_settings(), "predicted", regions=("WT",))
    assert out["rows"] == [] and "empty region" in out["failures"][0]["error"]


def test_compare_and_export(tmp_path):
    ref = [{"region": "WT", "modality": "t1", "feature_class": "firstorder", "feature_name": "Mean", "value": 10.0},
           {"region": "WT", "modality": "t1", "feature_class": "firstorder", "feature_name": "Zero", "value": 0.0}]
    pred = [dict(ref[0], value=11.0), dict(ref[1], value=2.0)]
    c = compare_radiomics_features(pred, ref).set_index("feature_name")
    assert c.loc["Mean", "abs_diff"] == 1 and c.loc["Mean", "rel_diff_percent"] == pytest.approx(10.0) and np.isnan(c.loc["Zero", "rel_diff_percent"])
    p = save_radiomics_csv(ref, tmp_path / "o.csv"); assert p.read_text().startswith("region")
    x = save_radiomics_excel(ref, tmp_path / "o.xlsx", {"setting": {"binWidth": 25}}); assert x.exists()
