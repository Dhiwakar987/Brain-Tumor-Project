import importlib.util, json, sys
from pathlib import Path
import pytest
from conftest import upload_files

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backup_restore.py"


def load_script():
    spec = importlib.util.spec_from_file_location("backup_restore", SCRIPT); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_backup_verify_restore_and_consistency(app, client, tmp_path, monkeypatch):
    client.post("/analysis/upload", data=upload_files(), content_type="multipart/form-data", headers={"X-Requested-With": "fetch"})
    m = load_script(); C = m.Config
    for k in ("DATABASE_PATH", "UPLOAD_DIR", "OUTPUT_DIR"): monkeypatch.setattr(C, k, app.config[k])
    monkeypatch.setattr(C, "BACKUP_DIR", tmp_path / "bk")
    z = m.backup(); v = m.verify(z)
    assert v["ok"] and v["n_exams"] == 1 and v["integrity"] == "ok"
    tgt = tmp_path / "restored"; m.restore(z, tgt)
    assert (tgt / "app.db").is_file() and any((tgt / "uploads").rglob("t1.nii.gz"))
    with pytest.raises(SystemExit): m.restore(z, tgt)  # refuses to overwrite
    # consistency check detects a deleted file
    next((app.config["UPLOAD_DIR"]).rglob("t2.nii.gz")).unlink()
    assert not m._db_check(app.config["DATABASE_PATH"], app.config["UPLOAD_DIR"], app.config["OUTPUT_DIR"])["ok"]
