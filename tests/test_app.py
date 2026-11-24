import io
import pytest
from conftest import make_labels, to_bytes, upload_files, wait_job

FETCH = {"X-Requested-With": "fetch"}


def up(client, **kw):
    return client.post(
        "/analysis/upload",
        data=upload_files(**kw),
        content_type="multipart/form-data",
        headers=FETCH,
    )


def test_startup_health_and_database(app, client):
    h = client.get("/health").get_json()
    assert h["status"] == "ok" and h["database"] is True and h["checkpoint_present"] is True


@pytest.mark.parametrize("path", ["/", "/dashboard", "/analysis/new", "/history", "/about"])
def test_static_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_unknown_routes_404(client):
    assert client.get("/analysis/999/results").status_code == 404
    assert client.get("/nope").status_code == 404
    assert client.post("/analysis/999/segment").status_code == 404


def test_2d_demo_sample_catalog_requires_matching_image_and_mask(tmp_path):
    from app import _list_2d_demo_samples

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for name in (
        "BRATS_001_slice010.npy",
        "BRATS_001_slice011.npy",
        "BRATS_002_slice001.npy",
        "not_a_slice.npy",
    ):
        (images_dir / name).touch()
    (masks_dir / "BRATS_001_slice010.npy").touch()
    (masks_dir / "mask_without_image.npy").touch()

    assert _list_2d_demo_samples(tmp_path) == {
        "BRATS_001": {10: "BRATS_001_slice010.npy"}
    }


def test_valid_upload_and_history(client):
    r = up(client)
    assert r.status_code == 200 and r.get_json()["ok"]
    assert client.get(r.get_json()["redirect"]).status_code == 200
    page = client.get("/history").get_data(as_text=True)
    assert "validated" in page
    d = client.get("/dashboard").get_data(as_text=True)
    assert ">1<" in d  # DB-driven count


def test_missing_modality(client):
    r = up(client, skip=("t2",))
    assert r.status_code == 400 and "Missing T2" in r.get_json()["errors"][0]
    assert "No examinations yet" in client.get("/history").get_data(as_text=True)


def test_invalid_nifti_rejected(client):
    r = up(client, corrupt="flair")
    assert r.status_code == 400 and not r.get_json()["ok"]


def test_wrong_extension_rejected(client):
    d = upload_files()
    d["t1"] = (io.BytesIO(b"MZ\x90\x00 executable"), "evil.exe")
    r = client.post(
        "/analysis/upload",
        data=d,
        content_type="multipart/form-data",
        headers=FETCH,
    )
    assert r.status_code == 400 and ".nii" in r.get_json()["errors"][0]


def test_executable_renamed_to_nii_rejected(client):
    d = upload_files()
    d["t1"] = (io.BytesIO(b"MZ\x90\x00" + b"\0" * 600), "evil.nii")
    assert client.post(
        "/analysis/upload",
        data=d,
        content_type="multipart/form-data",
        headers=FETCH,
    ).status_code == 400


def test_geometry_mismatch_rejected(client):
    r = up(client, mismatch="t1ce")
    assert r.status_code == 400 and "shape" in r.get_json()["errors"][0]


def test_path_traversal_filename_is_sanitised(app, client):
    d = upload_files()
    d["t1"] = (d["t1"][0], "../../evil.nii.gz")
    r = client.post(
        "/analysis/upload",
        data=d,
        content_type="multipart/form-data",
        headers=FETCH,
    )
    assert r.status_code == 200
    assert not (app.config["UPLOAD_DIR"].parent / "evil.nii.gz").exists()
    assert not list(app.config["UPLOAD_DIR"].parent.rglob("../evil*"))


def test_ground_truth_with_bad_labels_rejected(client):
    import numpy as np
    import nibabel as nib

    d = upload_files()
    lab = np.zeros((48, 48, 6), np.uint8)
    lab[5:9, 5:9, 1:3] = 3
    d["ground_truth"] = (
        io.BytesIO(to_bytes(nib.Nifti1Image(lab, np.eye(4)))),
        "gt.nii.gz",
    )
    r = client.post(
        "/analysis/upload",
        data=d,
        content_type="multipart/form-data",
        headers=FETCH,
    )
    assert r.status_code == 400 and "unexpected label" in r.get_json()["errors"][0]


def test_csrf_enforced_when_enabled(tmp_path, synthetic_checkpoint):
    from app import create_app
    from config import Config

    class C(Config):
        TESTING = False
        WTF_CSRF_ENABLED = True
        SECRET_KEY = "x"
        UPLOAD_DIR = tmp_path / "u"
        OUTPUT_DIR = tmp_path / "o"
        DATABASE_PATH = tmp_path / "d.db"
        CHECKPOINT_PATH = synthetic_checkpoint

    c = create_app(C).test_client()
    assert c.post("/analysis/upload", data={}, headers=FETCH).status_code == 400


def test_checkpoint_missing_gives_clear_failure(tmp_path):
    from app import create_app
    from config import Config

    class C(Config):
        WTF_CSRF_ENABLED = False
        UPLOAD_DIR = tmp_path / "u"
        OUTPUT_DIR = tmp_path / "o"
        DATABASE_PATH = tmp_path / "d.db"
        CHECKPOINT_PATH = tmp_path / "missing.pth"
        DEVICE = "cpu"

    app = create_app(C)
    c = app.test_client()
    assert c.get("/health").get_json()["checkpoint_present"] is False
    eid = up(c).get_json()["redirect"].split("/")[2]
    c.post(f"/analysis/{eid}/segment")
    s = wait_job(app, c, eid)
    assert s["job"]["status"] == "failed" and "not found" in s["job"]["error"] and s["exam_status"] == "failed"
    assert "/" not in s["job"]["error"].split("not found")[0][-3:]


def test_evaluate_without_ground_truth_refused(client):
    eid = up(client).get_json()["redirect"].split("/")[2]
    r = client.post(f"/analysis/{eid}/evaluate", headers=FETCH)
    assert r.status_code == 409 and "cannot be calculated" in r.get_json()["error"]
    assert "cannot be calculated" in client.get(
        f"/analysis/{eid}/evaluate"
    ).get_data(as_text=True)


@pytest.mark.slow
def test_full_workflow_synthetic(app, client):
    """Upload -> segment -> radiomics -> evaluate -> viewer -> report -> history -> delete (SYNTHETIC data, random weights)."""
    eid = up(client, with_gt=True).get_json()["redirect"].split("/")[2]

    assert client.post(
        f"/analysis/{eid}/radiomics",
        data={"source": "predicted"},
        headers=FETCH,
    ).status_code == 409  # needs segmentation

    assert client.post(f"/analysis/{eid}/segment").status_code == 202
    assert client.post(f"/analysis/{eid}/segment").status_code in (409, 202)

    s = wait_job(app, client, eid)
    assert s["job"]["status"] == "completed" and s["exam_status"] == "segmented"

    page = client.get(f"/analysis/{eid}/results").get_data(as_text=True)
    assert "Region volumes" in page and "Whole tumour" in page and "Tumour measurements" in page

    sl = client.get(
        f"/analysis/{eid}/slice?view=coronal&modality=t1&region=WT&source=predicted"
    ).get_json()
    assert sl["n_slices"] == 48 and len(sl["image"]) == 6
    assert client.get(f"/analysis/{eid}/slice?view=bogus").status_code == 400
    assert client.get(f"/analysis/{eid}/download/prediction").status_code == 200

    # evaluation
    client.post(f"/analysis/{eid}/evaluate")
    assert wait_job(app, client, eid)["job"]["status"] == "completed"
    assert "Dice" in client.get(
        f"/analysis/{eid}/evaluate"
    ).get_data(as_text=True)

    # radiomics on GT mask + export
    client.post(
        f"/analysis/{eid}/radiomics",
        data={
            "source": "ground_truth",
            "regions": ["WT"],
            "modalities": ["t1"],
        },
        headers=FETCH,
    )
    j = wait_job(app, client, eid)
    assert j["job"]["status"] == "completed"

    csv = client.get(
        f"/analysis/{eid}/radiomics/download?source=ground_truth"
    )
    assert csv.status_code == 200 and b"feature_name" in csv.data
    csv.close()

    xlsx = client.get(
        f"/analysis/{eid}/radiomics/download?source=ground_truth&format=xlsx"
    )
    assert xlsx.status_code == 200
    xlsx.close()