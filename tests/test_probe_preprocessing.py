"""Plumbing test of the probe on SYNTHETIC uint8 slices + masks and the (random-weight) synthetic checkpoint. Dice values are meaningless here."""
import importlib.util
from pathlib import Path
import numpy as np, pytest

S = Path(__file__).resolve().parents[1] / "scripts" / "probe_preprocessing.py"
spec = importlib.util.spec_from_file_location("probe", S); pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)


def test_mask_interpretations():
    assert pr.mask_to_classes(np.array([[0, 4], [2, 1]]))[0].tolist() == [[0, 3], [2, 1]]
    assert pr.mask_to_classes(np.array([[0, 3], [2, 1]]))[0].tolist() == [[0, 3], [2, 1]]
    assert pr.mask_to_classes(np.array([[0, 255]]))[0].tolist() == [[0, 1]]
    oh = np.zeros((4, 2, 2)); oh[2, 0, 0] = 1; oh[0] = 1 - oh[2]; assert pr.mask_to_classes(oh)[0][0, 0] == 2
    with pytest.raises(ValueError): pr.mask_to_classes(np.array([[0, 7]]))


def test_dice_none_when_both_empty():
    z = np.zeros((4, 4), bool); assert pr.dice(z, z) is None and pr.dice(~z, ~z) == 1.0


@pytest.mark.slow
def test_probe_runs_and_reports_all_stages(tmp_path, synthetic_checkpoint):
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir(); rng = np.random.default_rng(0)
    for i in range(3):
        np.save(tmp_path / "images" / f"s{i}.npy", rng.integers(0, 255, (4, 64, 64), dtype=np.uint8))
        m = np.zeros((64, 64), np.uint8); m[20:40, 20:40] = 2; m[25:30, 25:30] = 4; np.save(tmp_path / "masks" / f"s{i}.npy", m)
    rep = pr.main(["--images", str(tmp_path / "images"), "--masks", str(tmp_path / "masks"), "--checkpoint", str(synthetic_checkpoint), "--out", str(tmp_path / "out"), "--limit", "3"])
    assert rep["n_slices_used"] == 3 and len(rep["stage1_preprocessing_candidates"]) == 4 and len(rep["stage2_channel_orders_for_best_preprocessing"]) == 6
    assert (tmp_path / "out" / "PREPROCESSING_PROBE.md").is_file() and "BraTS labels" in rep["mask_interpretation"]


def test_probe_rejects_mismatched_image_mask_sizes(tmp_path, synthetic_checkpoint):
    (tmp_path / "i").mkdir(); (tmp_path / "m").mkdir()
    np.save(tmp_path / "i" / "a.npy", np.zeros((4, 64, 64), np.uint8)); m = np.zeros((32, 32), np.uint8); m[5:20, 5:20] = 1; np.save(tmp_path / "m" / "a.npy", m)
    with pytest.raises(SystemExit) as e:
        pr.main(["--images", str(tmp_path / "i"), "--masks", str(tmp_path / "m"), "--checkpoint", str(synthetic_checkpoint), "--out", str(tmp_path / "o")])
    assert "do not correspond" in str(e.value)


@pytest.mark.slow
def test_probe_verdict_warns_on_few_slices_and_random_weights(tmp_path, synthetic_checkpoint):
    (tmp_path / "i").mkdir(); (tmp_path / "m").mkdir(); rng = np.random.default_rng(1)
    for k in range(2):
        np.save(tmp_path / "i" / f"s{k}.npy", rng.integers(0, 255, (4, 64, 64), dtype=np.uint8)); m = np.zeros((64, 64), np.uint8); m[10:40, 10:40] = 1; np.save(tmp_path / "m" / f"s{k}.npy", m)
    rep = pr.main(["--images", str(tmp_path / "i"), "--masks", str(tmp_path / "m"), "--checkpoint", str(synthetic_checkpoint), "--out", str(tmp_path / "o")])
    text = " ".join(rep["verdicts"]); assert "too few" in text and ("NOT DECIDED" in text or "CLEAR" in text)
    assert "## Verdict" in (tmp_path / "o" / "PREPROCESSING_PROBE.md").read_text(encoding="utf-8")
