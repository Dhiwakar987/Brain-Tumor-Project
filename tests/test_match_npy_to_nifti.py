"""Recovery test on SYNTHETIC data: build .npy slices from synthetic NIfTI with a KNOWN channel order / orientation / scaling and check the tool recovers them."""
import importlib.util
from pathlib import Path
import nibabel as nib, numpy as np, pytest
from scipy import ndimage as ndi

S = Path(__file__).resolve().parents[1] / "scripts" / "match_npy_to_nifti.py"
spec = importlib.util.spec_from_file_location("match_npy", S); mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)
STORED = ["flair", "t1", "t1ce", "t2"]  # the known truth we plant


def build(tmp_path, pid="BraTS20_Training_001", op="rot90", slices=(8, 10, 12, 14)):
    rng = np.random.default_rng(0); shape = (48, 48, 20); d = tmp_path / "nifti" / pid; d.mkdir(parents=True)
    brain = ndi.gaussian_filter((np.indices(shape)[0] - 24) ** 2 + (np.indices(shape)[1] - 24) ** 2 < 18 ** 2, 1).astype(float)
    def tex():  # independent, structured texture per modality (like real sequences, which differ)
        g = ndi.gaussian_filter(rng.random(shape), 1.2); return (g - g.mean()) / g.std()
    vols = {m: ((brain * (100 + 40 * tex())) * (brain > 0.2)).clip(1).astype(np.float32) * (brain > 0.2) for m in ("t1", "t1ce", "t2", "flair")}
    for m, v in vols.items(): nib.save(nib.Nifti1Image(v, np.eye(4)), str(d / f"{pid}_{m}.nii.gz"))
    seg = np.zeros(shape, np.uint8); seg[18:30, 18:30, 6:18] = 2; seg[22:26, 22:26, 8:16] = 4; nib.save(nib.Nifti1Image(seg, np.eye(4)), str(d / f"{pid}_seg.nii.gz"))
    (tmp_path / "images").mkdir(exist_ok=True)
    for i in slices:
        ch = []
        for m in STORED:
            s = mn.OPS[op](np.take(vols[m], i, axis=2)); ch.append(np.round((s - s.min()) / (s.max() - s.min()) * 255))
        np.save(tmp_path / "images" / f"{pid}_slice{i}.npy", np.stack(ch).astype(np.uint8))
    return seg


def test_recovers_channel_order_orientation_scaling_and_masks(tmp_path):
    seg = build(tmp_path)
    r = mn.main(["--nifti-root", str(tmp_path / "nifti"), "--images", str(tmp_path / "images"), "--write-masks", str(tmp_path / "masks"), "--out", str(tmp_path / "out")])
    d = r["details"][0]
    assert r["mapping_determined"] and d["channel_order"] == STORED and d["slice_axis"] == 2 and d["orientation"] == "rot90"
    assert d["scaling"].startswith("per-slice min-max, round") and "per-volume" not in d["scaling"] and r["masks_written"] == 4
    m = np.load(tmp_path / "masks" / "BraTS20_Training_001_slice10.npy"); assert np.array_equal(m, np.rot90(seg[:, :, 10], 1)) and set(np.unique(m)) <= {0, 2, 4}


def test_refuses_when_stored_data_does_not_come_from_these_volumes(tmp_path):
    build(tmp_path)
    for f in (tmp_path / "images").glob("*.npy"): np.save(f, np.random.default_rng(5).integers(0, 255, (4, 48, 48), dtype=np.uint8))
    r = mn.main(["--nifti-root", str(tmp_path / "nifti"), "--images", str(tmp_path / "images"), "--write-masks", str(tmp_path / "masks"), "--out", str(tmp_path / "out")])
    assert not r["mapping_determined"] and r["masks_written"] == 0 and not (tmp_path / "masks").exists()
