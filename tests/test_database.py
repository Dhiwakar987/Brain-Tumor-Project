import pytest
from database import db


@pytest.fixture()
def d(tmp_path):
    db.initialize_database(tmp_path / "t.db"); return db


def test_init_creates_tables(d):
    with d.connect() as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"examinations", "uploaded_files", "segmentation_results", "radiomics_results", "evaluation_results", "generated_reports", "application_settings"} <= names


def test_create_get_update_delete_cascade(d):
    eid, uid = d.create_examination("lbl", True, "m", "abc", {"a": 1})
    d.save_uploaded_file_record(eid, "t1", "x.nii", "t1.nii", 10, "h")
    d.update_examination_status(eid, status="validated", warnings_json=["w"])
    e = d.get_examination(eid)
    assert e["status"] == "validated" and e["warnings"] == ["w"] and len(e["files"]) == 1 and e["config_snapshot"] == {"a": 1}
    d.save_radiomics_result(eid, "predicted", [{"region": "WT", "modality": "t1", "feature_class": "shape", "feature_name": "x", "value": 1.0}], {"s": 1}, "v", [], {})
    assert len(d.get_radiomics_rows(eid)) == 1
    assert d.delete_examination(eid) == uid
    assert d.get_examination(eid) is None
    with d.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM uploaded_files").fetchone()[0] == 0 and c.execute("SELECT COUNT(*) FROM radiomics_results").fetchone()[0] == 0


def test_dashboard_stats_empty_and_filled(d):
    assert d.dashboard_stats()["total_exams"] == 0
    d.create_examination(); assert d.dashboard_stats()["total_exams"] == 1


def test_unsupported_field_rejected(d):
    eid, _ = d.create_examination()
    with pytest.raises(ValueError): d.update_examination_status(eid, status_x="1") if False else d.update_examination_status(eid, bogus_json="x")


def test_duplicate_job_guard_and_restart_recovery(d):
    eid, _ = d.create_examination()
    j = d.start_job(eid, "segmentation"); assert j
    assert d.start_job(eid, "segmentation") is None  # duplicate submission blocked
    d.update_examination_status(eid, status="processing")
    assert d.mark_interrupted_jobs() == 1
    assert d.get_latest_job(eid)["status"] == "interrupted" and d.get_examination(eid)["status"] == "interrupted"
    assert d.start_job(eid, "segmentation")  # can restart after interruption


def test_settings_roundtrip(d):
    d.set_setting("k", "v"); d.set_setting("k", "v2"); assert d.get_setting("k") == "v2" and d.get_setting("zz", "dflt") == "dflt"
