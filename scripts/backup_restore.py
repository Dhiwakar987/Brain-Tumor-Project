"""Backup / restore / consistency check.
  python scripts/backup_restore.py backup            -> backups/backup_<ts>.zip (SQLite online backup + outputs + uploads) + verification
  python scripts/backup_restore.py verify  FILE.zip  -> checks zip integrity, DB integrity and DB<->file consistency inside the zip
  python scripts/backup_restore.py restore FILE.zip --target DIR   -> restores into an EMPTY/new directory (never overwrites existing data)
  python scripts/backup_restore.py check             -> detect missing files / orphan folders in the live installation"""
import argparse, json, shutil, sqlite3, sys, tempfile, zipfile
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config


def _db_check(db_path, root_up, root_out):
    c = sqlite3.connect(db_path); c.row_factory = sqlite3.Row
    res = {"integrity": c.execute("PRAGMA integrity_check").fetchone()[0], "missing_files": [], "n_exams": 0}
    for e in c.execute("SELECT id, exam_uid FROM examinations"):
        res["n_exams"] += 1
        for f in c.execute("SELECT stored_name FROM uploaded_files WHERE examination_id=?", (e["id"],)):
            if not (root_up / e["exam_uid"] / f["stored_name"]).is_file(): res["missing_files"].append(f"{e['id']}/{f['stored_name']}")
        for s in c.execute("SELECT prediction_file FROM segmentation_results WHERE examination_id=?", (e["id"],)):
            if not (root_out / e["exam_uid"] / s["prediction_file"]).is_file(): res["missing_files"].append(f"{e['id']}/{s['prediction_file']}")
    c.close(); res["ok"] = res["integrity"] == "ok" and not res["missing_files"]; return res


def backup():
    Config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    out = Config.BACKUP_DIR / f"backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with tempfile.TemporaryDirectory() as t:
        snap = Path(t) / "app.db"
        src = sqlite3.connect(Config.DATABASE_PATH); dst = sqlite3.connect(snap); src.backup(dst); dst.close(); src.close()  # consistent online snapshot
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(snap, "app.db")
            for name, base in (("uploads", Config.UPLOAD_DIR), ("outputs", Config.OUTPUT_DIR)):
                for p in base.rglob("*"):
                    if p.is_file() and p.resolve() != Path(Config.DATABASE_PATH).resolve() and not p.name.startswith(("partial_", "app.db")):
                        z.write(p, f"{name}/{p.relative_to(base)}")
    print(json.dumps({"backup": out.name, "verification": verify(out)}, indent=1)); return out


def verify(zpath):
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        with tempfile.TemporaryDirectory() as t:
            z.extractall(t); r = _db_check(Path(t) / "app.db", Path(t) / "uploads", Path(t) / "outputs")
    r["zip_integrity"] = "ok" if bad is None else f"corrupt member {bad}"; r["ok"] = r["ok"] and bad is None; return r


def restore(zpath, target):
    target = Path(target)
    if target.exists() and any(target.iterdir()):
        sys.exit("Refusing to restore into a non-empty directory (never overwrites existing data). Choose a new --target.")
    v = verify(zpath)
    if not v["ok"]: sys.exit(f"Backup failed verification, not restoring: {v}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z: z.extractall(target)
    print(f"Restored into {target}. Point DATABASE_PATH={target/'app.db'}, UPLOAD_DIR={target/'uploads'}, OUTPUT_DIR={target/'outputs'} to use it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["backup", "verify", "restore", "check"]); ap.add_argument("file", nargs="?"); ap.add_argument("--target")
    a = ap.parse_args()
    if a.cmd == "backup": backup()
    elif a.cmd == "verify": print(json.dumps(verify(a.file), indent=1))
    elif a.cmd == "restore": restore(a.file, a.target or sys.exit("--target required"))
    else: print(json.dumps(_db_check(Config.DATABASE_PATH, Config.UPLOAD_DIR, Config.OUTPUT_DIR), indent=1))
