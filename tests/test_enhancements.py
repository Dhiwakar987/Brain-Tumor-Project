"""Tests for measurements, 3-D mesh, three-panel comparison, exports and report extras (SYNTHETIC data)."""
import io, json
import numpy as np
import pytest
from conftest import upload_files, wait_job
from segmentation.measurements import region_measurements
from visualization.mesh3d import build_meshes
from visualization.mri_viewer import axis_labels

FETCH = {"X-Requested-With": "fetch"}


def sphere(shape=(48, 48, 48), r=10):
    x, y, z = np.indices(shape); c = np.array(shape) / 2
    return ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) <= r * r


def test_measurements_cube_uses_header_spacing():
    m = np.zeros((30, 30, 30), bool); m[5:15, 5:15, 5:15] = True
    r = region_measurements(m, (2.0, 1.0, 1.0))
    assert r["voxel_count"] == 1000 and r["volume_mm3"] == pytest.approx(2000) and r["volume_ml"] == pytest.approx(2.0)
    assert r["bbox_extent_mm"] == [20.0, 10.0, 10.0] and r["voxel_volume_mm3"] == 2.0


def test_measurements_sphere_sanity():
    r = region_measurements(sphere(), (1, 1, 1))
    assert 0.85 < r["sphericity"] < 1.1 and 18 <= r["max_3d_diameter_mm"] <= 22 and r["surface_area_mm2"] > 1000


def test_measurements_empty_and_invalid_spacing():
    r = region_measurements(np.zeros((5, 5, 5), bool), (1, 1, 1)); assert r["voxel_count"] == 0 and r["note"] == "empty region" and r["sphericity"] is None
    with pytest.raises(ValueError): region_measurements(np.ones((3, 3, 3), bool), (0, 1, 1))


def test_mesh_built_from_real_arrays_and_in_mm():
    img = np.zeros((40, 40, 40), np.float32); img[8:32, 8:32, 8:32] = 100.0
    lab = np.zeros((40, 40, 40), np.int16); lab[15:22, 15:22, 15:22] = 4
    m = build_meshes(img, lab, (2.0, 1.0, 1.0))
    assert m["brain"] and list(m["regions"]) == ["Enhancing (label 4)"]
    xs = np.array(m["regions"]["Enhancing (label 4)"]["x"])
    assert 28 <= xs.min() and xs.max() <= 46  # x is in mm: voxels 15..22 * 2.0 spacing
    assert build_meshes(img, np.zeros_like(lab), (1, 1, 1))["regions"] == {}  # no fabricated tumour


def test_axis_labels():
    assert axis_labels(("R", "A", "S"), "axial") == ("L \u2190 \u2192 R", "P \u2190 \u2192 A")
    assert axis_labels(("L", "P", "S"), "coronal")[1] == "I \u2190 \u2192 S"


@pytest.mark.slow
def test_comparison_mesh_measurements_exports_and_report(app, client):
    r = client.post("/analysis/upload", data=upload_files(with_gt=True), content_type="multipart/form-data", headers=FETCH); eid = r.get_json()["redirect"].split("/")[2]
    assert client.get(f"/analysis/{eid}/compare-slice").status_code == 400 and client.get(f"/analysis/{eid}/mesh3d").status_code == 400  # before segmentation: no fabricated output
    assert client.get(f"/analysis/{eid}/measurements").status_code == 404
    client.post(f"/analysis/{eid}/segment"); assert wait_job(app, client, eid)["job"]["status"] == "completed"
    c = client.get(f"/analysis/{eid}/compare-slice?view=axial&modality=t1&region=WT&index=3").get_json()
    n = len(c["image"]); assert n and len(c["prediction"]) == n and len(c["ground_truth"]) == n and c["xlabel"] and 0 <= c["index"] < c["n_slices"]
    gt = np.array([[np.nan if v is None else v for v in row] for row in c["ground_truth"]], float)
    assert np.nansum(gt) > 0  # real GT tumour appears on the chosen slice
    assert client.get(f"/analysis/{eid}/compare-slice?view=bad").status_code == 400
    me = client.get(f"/analysis/{eid}/measurements").get_json(); assert set(me["predicted"]) == {"WT", "TC", "ET"} and me["ground_truth"]["WT"]["voxel_count"] > 0
    assert client.get(f"/analysis/{eid}/mesh3d?modality=t1").status_code == 200
    # radiomics (GT) so exports contain radiomics
    client.post(f"/analysis/{eid}/radiomics", data={"source": "ground_truth", "regions": ["WT"], "modalities": ["t1"]}, headers=FETCH); assert wait_job(app, client, eid)["job"]["status"] == "completed"
    client.post(f"/analysis/{eid}/evaluate"); wait_job(app, client, eid)
    j = client.get(f"/analysis/{eid}/export?format=json").get_json()
    assert j["summary"]["classification"].startswith("not available") and j["measurements"] and j["radiomics"] and j["evaluation"]
    # exported radiomics values == DB values
    from database import db
    assert sorted(r["value"] for r in j["radiomics"]) == sorted(r["value"] for r in db.get_radiomics_rows(int(eid)))
    import pandas as pd
    x = client.get(f"/analysis/{eid}/export?format=xlsx"); assert x.status_code == 200
    sheets = pd.read_excel(io.BytesIO(x.data), sheet_name=None); assert {"summary", "measurements", "radiomics", "evaluation"} <= set(sheets) and len(sheets["radiomics"]) == len(j["radiomics"])
    csv = client.get(f"/analysis/{eid}/export?format=csv"); assert csv.status_code == 200 and b"radiomics:" in csv.data and b"volume_mm3" in csv.data
    assert client.get(f"/analysis/{eid}/export?format=exe").status_code == 400
    client.post(f"/analysis/{eid}/report/generate"); pdf = client.get(f"/analysis/{eid}/download-report").data
    assert pdf[:5] == b"%PDF-" and len(pdf) > 8000
    for p in ("results", "radiomics", "evaluate", "report"): assert client.get(f"/analysis/{eid}/{p}").status_code == 200
    assert "three-panel" in client.get(f"/analysis/{eid}/results").get_data(as_text=True).lower()


def test_no_gt_comparison_unavailable_message(app, client):
    eid = client.post("/analysis/upload", data=upload_files(), content_type="multipart/form-data", headers=FETCH).get_json()["redirect"].split("/")[2]
    client.post(f"/analysis/{eid}/segment"); wait_job(app, client, eid)
    assert client.get(f"/analysis/{eid}/compare-slice").get_json()["ground_truth"] is None
    assert "comparison is unavailable" in client.get(f"/analysis/{eid}/results").get_data(as_text=True)


# ---------------- v2: noise detection, new pages, performance page, Windows-safe IO ----------------
def test_random_noise_volume_is_flagged_not_trusted():
    from model.preprocessing import spatial_smoothness
    rng = np.random.default_rng(0)
    assert spatial_smoothness(rng.random((32, 32, 32))) < 0.1               # like the uploaded brain_image file
    from scipy import ndimage as ndi
    smooth = ndi.gaussian_filter(rng.random((32, 32, 32)), 3); assert spatial_smoothness(smooth) > 0.8  # structured data


def test_upload_of_noise_gives_warning(client):
    """Pure uniform noise in every voxel (like the user's uploaded brain_image file) must be flagged."""
    import nibabel as nib
    from conftest import to_bytes
    rng = np.random.default_rng(1)
    d = {k: (io.BytesIO(to_bytes(nib.Nifti1Image(rng.random((32, 32, 32)).astype(np.float32), np.eye(4)))), f"{k}.nii.gz") for k in ("t1", "t1ce", "t2", "flair")}
    r = client.post("/analysis/upload", data=d, content_type="multipart/form-data", headers=FETCH).get_json()
    assert r["ok"] and sum("random noise" in w for w in r["warnings"]) == 4


@pytest.mark.parametrize("path", ["/classification", "/explainability", "/performance"])
def test_new_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_classification_and_xai_pages_never_show_predictions(client):
    c = client.get("/classification").get_data(as_text=True); x = client.get("/explainability").get_data(as_text=True)
    assert "unavailable" in c and "confidence" not in c.split("What is needed")[0].lower().replace("or confidence", "")
    assert "not implemented" in x and "<img" not in x


@pytest.mark.slow
def test_performance_page_empty_then_filled(app, client):
    assert "No evaluated cases yet" in client.get("/performance").get_data(as_text=True)
    eid = client.post("/analysis/upload", data=upload_files(with_gt=True), content_type="multipart/form-data", headers=FETCH).get_json()["redirect"].split("/")[2]
    client.post(f"/analysis/{eid}/segment"); wait_job(app, client, eid); client.post(f"/analysis/{eid}/evaluate"); wait_job(app, client, eid)
    page = client.get("/performance").get_data(as_text=True)
    assert "n = 1" in page and "not</b> an independent test" in page.replace("<b>not</b>", "<b>not</b>")
    from database import db
    ps = db.performance_summary(); assert ps["n_cases"] == 1 and set(ps["regions"]) == {"WT", "TC", "ET"}


def test_uncompressed_nii_does_not_hold_file_handle(tmp_path):
    """Windows cannot delete a file that is still memory-mapped; load_nifti must copy the data."""
    import nibabel as nib
    from conftest import make_volume
    from segmentation.nifti_io import load_nifti
    p = tmp_path / "v.nii"; nib.save(make_volume(), str(p)); v = load_nifti(p)
    assert not isinstance(v.data, np.memmap) and not isinstance(getattr(v.data, "base", None), np.memmap)
    p.unlink(); assert not p.exists()


def test_dotenv_with_bom_and_quotes_is_parsed(tmp_path, monkeypatch):
    """Windows Notepad saves .env with a UTF-8 BOM; the first variable must still load."""
    import importlib, config
    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbfNEURO_TEST_FIRST=abc\r\nNEURO_TEST_Q=\"quoted value\"  # comment\r\n")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path); monkeypatch.delenv("NEURO_TEST_FIRST", raising=False); monkeypatch.delenv("NEURO_TEST_Q", raising=False)
    config._load_dotenv()
    import os; assert os.environ["NEURO_TEST_FIRST"] == "abc" and os.environ["NEURO_TEST_Q"] == "quoted value"
    os.environ.pop("NEURO_TEST_FIRST", None); os.environ.pop("NEURO_TEST_Q", None)


def test_sqlite_journal_mode_avoids_wal_in_onedrive(tmp_path, monkeypatch):
    from database import db
    monkeypatch.delenv("SQLITE_JOURNAL_MODE", raising=False)
    d = tmp_path / "OneDrive" / "x"; d.mkdir(parents=True); db.initialize_database(d / "t.db")
    with db.connect() as c: assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    db.initialize_database(tmp_path / "plain.db")
    with db.connect() as c: assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
