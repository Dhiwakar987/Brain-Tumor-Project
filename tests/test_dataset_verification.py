"""Dataset verifier tested on a SYNTHETIC BraTS-style folder (portable tmp paths)."""
import importlib.util
from pathlib import Path
import nibabel as nib, numpy as np
from conftest import make_labels, make_volume
from scipy import ndimage as ndi

S = Path(__file__).resolve().parents[1] / "scripts" / "verify_dataset.py"
spec = importlib.util.spec_from_file_location("verify_dataset", S); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)


def smooth_vol(seed, shape=(40, 40, 12)):
    rng = np.random.default_rng(seed); d = ndi.gaussian_filter(rng.random(shape), 2).astype(np.float32) + 0.1
    return nib.Nifti1Image(d, np.eye(4))


def make_patient(root, pid, skip=(), bad_shape=None, corrupt=None):
    d = root / pid; d.mkdir(parents=True)
    for i, r in enumerate(("t1", "t1ce", "t2", "flair")):
        if r in skip: continue
        img = smooth_vol(i, (40, 40, 14) if bad_shape == r else (40, 40, 12)); nib.save(img, str(d / f"{pid}_{r}.nii.gz"))
    if "seg" not in skip: nib.save(make_labels((40, 40, 12)), str(d / f"{pid}_seg.nii.gz"))
    if corrupt: (d / f"{pid}_{corrupt}.nii.gz").write_bytes(b"not a nifti")


def test_classify_names():
    assert vd.classify("BraTS20_Training_001_t1ce.nii.gz") == "t1ce" and vd.classify("X_t1.nii") == "t1" and vd.classify("X_T1GD.nii.gz") == "t1ce"
    assert vd.classify("brain_image.nii") is None  # modality cannot be identified from name -> not guessed


def test_complete_missing_mismatched_corrupt_and_noise(tmp_path):
    root = tmp_path / "ds"; root.mkdir()
    make_patient(root, "P001"); make_patient(root, "P002", skip=("flair", "seg")); make_patient(root, "P003", bad_shape="t2"); make_patient(root, "P004", corrupt="t1")
    d = root / "P005"; d.mkdir(); nib.save(nib.Nifti1Image(np.random.default_rng(0).random((32, 32, 32)).astype(np.float32), np.eye(4)), str(d / "brain_image.nii.gz"))
    rows = {r["patient"]: r for r in vd.main([str(root), "--out", str(tmp_path / "out")])}
    assert rows["P001"]["usable"] == "YES" and rows["P001"]["seg_labels"] == [0, 1, 2, 4]
    assert "missing: FLAIR, SEG" in rows["P002"]["issues"][0] and rows["P002"]["usable"] == "NO"
    assert any("shape" in i for i in rows["P003"]["issues"])
    assert rows["P004"]["t1_ok"] == "CORRUPT"
    assert rows["P005"]["unknown_files"] == ["brain_image.nii.gz"] and rows["P005"]["usable"] == "NO"
    assert (tmp_path / "out" / "dataset_verification.csv").is_file() and "P001" in (tmp_path / "out" / "DATASET_VERIFICATION.md").read_text()
