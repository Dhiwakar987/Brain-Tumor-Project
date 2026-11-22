import pytest
from conftest import upload_files, wait_job


@pytest.mark.slow
def test_report_generation_from_saved_results(app, client):
    r = client.post("/analysis/upload", data=upload_files(with_gt=True), content_type="multipart/form-data", headers={"X-Requested-With": "fetch"})
    eid = r.get_json()["redirect"].split("/")[2]
    client.post(f"/analysis/{eid}/segment"); assert wait_job(app, client, eid)["job"]["status"] == "completed"
    client.post(f"/analysis/{eid}/evaluate"); assert wait_job(app, client, eid)["job"]["status"] == "completed"
    assert client.post(f"/analysis/{eid}/report/generate").status_code == 302
    pdf = client.get(f"/analysis/{eid}/download-report")
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-" and len(pdf.data) > 3000
    meta = client.get(f"/analysis/{eid}/download/metadata").get_json()
    assert meta["checkpoint_sha256"] and len(meta["inputs"]) == 5 and meta["evaluation"] and meta["timings_s"]["report_generation_s"] >= 0
    assert "uploads" not in str(meta["inputs"])  # no private paths


def test_report_without_segmentation_states_not_available(app, client):
    r = client.post("/analysis/upload", data=upload_files(), content_type="multipart/form-data", headers={"X-Requested-With": "fetch"})
    eid = r.get_json()["redirect"].split("/")[2]
    client.post(f"/analysis/{eid}/report/generate")
    assert client.get(f"/analysis/{eid}/download-report").data[:5] == b"%PDF-"
