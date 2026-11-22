import numpy as np
from segmentation import metrics as mt
from segmentation.regions import (calculate_region_volume_ml, create_enhancing_tumor_mask, create_tumor_core_mask,
                                  create_whole_tumor_mask, get_region_voxel_count, validate_label_values)
import pytest


def cube(n=10, a=2, b=6):
    m = np.zeros((n, n, n), bool); m[a:b, a:b, a:b] = True; return m


def test_dice_iou_perfect():
    m = cube()
    assert mt.dice_coefficient(m, m) == 1.0 and mt.iou_score(m, m) == 1.0


def test_dice_iou_known_overlap():
    a = np.zeros(10, bool); a[:6] = True      # 6 voxels
    b = np.zeros(10, bool); b[3:9] = True     # 6 voxels, overlap 3
    assert mt.dice_coefficient(a, b) == pytest.approx(2 * 3 / 12)
    assert mt.iou_score(a, b) == pytest.approx(3 / 9)
    assert mt.precision_score(a, b) == pytest.approx(0.5) and mt.recall_score(a, b) == pytest.approx(0.5)
    assert mt.specificity_score(a, b) == pytest.approx(1 / 4)  # tn=1 (idx 9), fp=3 -> 1/4


def test_empty_both_is_undefined_not_zero_or_one():
    e = np.zeros((5, 5, 5), bool)
    r = mt.compute_binary_metrics(e, e)
    assert r["dice"] is None and r["iou"] is None and r["hd95_mm"] is None and r["precision"] is None and r["recall"] is None
    assert r["specificity"] == 1.0 and "both" in r["notes"]


def test_one_empty():
    e, m = np.zeros((6, 6, 6), bool), cube(6, 1, 4)
    r = mt.compute_binary_metrics(e, m)
    assert r["dice"] == 0.0 and r["iou"] == 0.0 and r["hd95_mm"] is None and r["precision"] is None and r["recall"] == 0.0


def test_hd95_zero_identical_and_spacing_units():
    m = cube()
    assert mt.hausdorff95(m, m) == 0.0
    shifted = np.roll(m, 2, axis=0)
    d1 = mt.hausdorff95(m, shifted, (1, 1, 1)); d2 = mt.hausdorff95(m, shifted, (2, 1, 1))
    assert d2 == pytest.approx(2 * d1, rel=0.2) and d1 > 0  # physical spacing is used


def test_region_definitions():
    lab = np.array([0, 1, 2, 4, 4])
    assert create_whole_tumor_mask(lab).sum() == 4 and create_tumor_core_mask(lab).sum() == 3 and create_enhancing_tumor_mask(lab).sum() == 2


def test_volume_ml_and_invalid_spacing():
    m = np.ones((10, 10, 10), bool)
    assert calculate_region_volume_ml(m, (1, 1, 1)) == pytest.approx(1.0)
    assert calculate_region_volume_ml(m, (2, 1, 1)) == pytest.approx(2.0)
    assert get_region_voxel_count(m) == 1000
    for bad in [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (1, 1)]:
        with pytest.raises(ValueError): calculate_region_volume_ml(m, bad)


def test_label_validation():
    assert validate_label_values(np.array([0, 1, 2, 4])) == []
    assert validate_label_values(np.array([0, 3])) == [3]


def test_evaluate_regions_keys():
    lab = np.zeros((8, 8, 8), np.int16); lab[2:6, 2:6, 2:6] = 2; lab[3:5, 3:5, 3:5] = 4
    r = mt.evaluate_regions(lab, lab, (1, 1, 1))
    assert set(r) == {"WT", "TC", "ET"} and all(v["dice"] == 1.0 for v in r.values())
