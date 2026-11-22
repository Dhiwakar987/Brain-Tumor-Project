from pathlib import Path
import numpy as np
import pytest
import nibabel as nib
from conftest import make_labels, make_volume
from config import Config
from model.checkpoint_validation import inspect_checkpoint, validate_checkpoint
from model.inference import map_prediction_labels, predict_slice, predict_volume
from model.model_loader import ModelLoadError, load_model
from model.preprocessing import normalize_mri, preprocess_mri, unpad_prediction, validate_modalities
from segmentation.nifti_io import NiftiError, load_nifti, save_nifti_like, verify_saved_geometry

CFG = Config.model_config()


def vols(shape=(48, 48, 6), **kw):
    """Four synthetic modalities saved/loaded through a portable temporary directory."""
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for i, m in enumerate(("t1", "t1ce", "t2", "flair")):
            p = Path(d) / f"{m}.nii.gz"; nib.save(make_volume(shape, seed=i, **kw), str(p)); out[m] = load_nifti(p)
    return out


def test_validate_modalities_ok_and_missing():
    v = vols(); assert validate_modalities(v)["ok"]
    v.pop("t2"); r = validate_modalities(v); assert not r["ok"] and "Missing" in r["errors"][0]


def test_validate_modalities_shape_and_affine_mismatch():
    v = vols(); other = vols((48, 48, 8)); v["flair"] = other["flair"]
    assert any("shape" in e for e in validate_modalities(v)["errors"])
    v = vols(); v2 = vols(zooms=(2.0, 2.0, 2.0)); v["t2"] = v2["t2"]
    assert any("affine" in e for e in validate_modalities(v)["errors"])


def test_validate_nan_and_zero():
    v = vols(); v["t1"].data[5, 5, 2] = np.nan
    assert any("NaN" in e for e in validate_modalities(v)["errors"])
    v = vols(); v["t1"].data[:] = 0
    assert any("entirely zero" in e for e in validate_modalities(v)["errors"])


def test_normalization_modes():
    v = vols()["t1"].data
    z, p = normalize_mri(v, "zscore_nonzero")
    assert abs(z[v != 0].mean()) < 1e-3 and (z[v == 0] == 0).all()
    m, _ = normalize_mri(v, "minmax_volume"); assert m.min() == 0 and m.max() == pytest.approx(1)
    n, _ = normalize_mri(v, "none"); assert np.allclose(n, v)
    with pytest.raises(ValueError): normalize_mri(v, "bogus")


def test_preprocess_pads_to_multiple_and_unpads():
    v = vols((50, 45, 4)); stack, rec = preprocess_mri(v, CFG)
    assert stack.shape[0] == 4 and stack.shape[1] % 32 == 0 and stack.shape[2] % 32 == 0
    cls = np.zeros(stack.shape[1:], np.uint8)
    assert unpad_prediction(cls, rec["pad_hw"]).shape == (50, 45, 4)


def test_label_mapping():
    pred = np.array([[0, 1], [2, 3]], np.uint8)
    out = map_prediction_labels(pred, {0: 0, 1: 1, 2: 2, 3: 4})
    assert out.tolist() == [[0, 1], [2, 4]]
    with pytest.raises(ValueError): map_prediction_labels(np.array([5], np.uint8), {0: 0, 1: 1})


def test_nifti_roundtrip_preserves_geometry(tmp_path):
    ref = tmp_path / "ref.nii.gz"; nib.save(make_volume(zooms=(1.5, 1.5, 3.0)), ref)
    arr = np.random.default_rng(0).integers(0, 5, (48, 48, 6))
    out = save_nifti_like(arr, ref, tmp_path / "o.nii.gz")
    assert verify_saved_geometry(out, ref) == {"shape_match": True, "affine_match": True, "readable": True}
    assert load_nifti(out, dtype=np.int16).zooms == (1.5, 1.5, 3.0) and not list(tmp_path.glob("partial_*"))


def test_invalid_nifti_raises(tmp_path):
    p = tmp_path / "bad.nii.gz"; p.write_bytes(b"nope")
    with pytest.raises(NiftiError): load_nifti(p)


def test_checkpoint_architecture_compat_synthetic(synthetic_checkpoint):
    rep = validate_checkpoint(synthetic_checkpoint, CFG)
    assert rep["checks"]["strict_load_ok"] and rep["compatibility"]["n_missing"] == 0
    assert rep["checks"]["real_inference_completed"] is False  # never claimed by this function
    info = inspect_checkpoint(synthetic_checkpoint); assert info["inferred_in_channels"] == 4 and info["inferred_num_classes"] == 4


def test_incompatible_checkpoint_rejected(tmp_path):
    import torch, segmentation_models_pytorch as smp
    bad = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)
    p = tmp_path / "bad.pth"; torch.save(bad.state_dict(), p)
    rep = validate_checkpoint(p, CFG); assert not rep["checks"]["strict_load_ok"] and rep["compatibility"]["n_shape_mismatches"] > 0
    with pytest.raises(ModelLoadError): load_model(p, CFG, "cpu")
    with pytest.raises(ModelLoadError): load_model(tmp_path / "missing.pth", CFG, "cpu")


@pytest.mark.slow
def test_prediction_output_shape(synthetic_checkpoint):
    model, dev = load_model(synthetic_checkpoint, CFG, "cpu")
    assert not model.training
    stack, rec = preprocess_mri(vols(), CFG)
    s = predict_slice(model, stack[..., 0], dev); assert s.shape == stack.shape[1:3]
    vol, stats = predict_volume(model, stack, dev, batch_size=3)
    assert vol.shape == stack.shape[1:] and vol.dtype == np.uint8 and stats["slices_predicted"] == 6 and max(vol.flatten()) <= 3


@pytest.mark.parametrize("name", ["identity", "transpose", "flipud", "fliplr", "rot90", "rot180", "rot270", "transpose+rot180"])
def test_orientation_roundtrip_nonsquare_and_matches_match_script(name):
    import importlib.util
    from model.preprocessing import apply_orientation, invert_orientation
    x = np.random.default_rng(0).random((4, 50, 40, 3))
    assert np.array_equal(invert_orientation(apply_orientation(x, name, (1, 2)), name, (1, 2)), x)
    spec = importlib.util.spec_from_file_location("mn", Path(__file__).resolve().parents[1] / "scripts" / "match_npy_to_nifti.py"); mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)
    sl = x[0, :, :, 0]; assert np.array_equal(apply_orientation(sl, name, (0, 1)), mn.OPS[name](sl))  # same definition in pipeline and matcher


def test_pipeline_with_orientation_keeps_nifti_grid(synthetic_checkpoint):
    import copy
    from segmentation.prediction import run_segmentation
    cfg = copy.deepcopy(CFG); cfg["preprocessing"]["slice_orientation"] = "rot90"
    v = vols((50, 40, 4)); import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "t1.nii.gz"); nib.save(nib.Nifti1Image(v["t1"].data, v["t1"].affine), ref)
        r = run_segmentation(v, ref, os.path.join(d, "p.nii.gz"), cfg, synthetic_checkpoint, "cpu", 2)
        assert r["transform"]["slice_orientation"] == "rot90" and load_nifti(os.path.join(d, "p.nii.gz")).shape == (50, 40, 4) and r["output_geometry_check"]["shape_match"]
