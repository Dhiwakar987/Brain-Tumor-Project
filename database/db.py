"""SQLite access layer. Large MRI volumes stay on disk; the DB stores paths/metadata only."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.sql")
_DB_PATH = None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure(db_path):
    global _DB_PATH
    _DB_PATH = Path(db_path)


@contextmanager
def connect():
    if _DB_PATH is None:
        raise RuntimeError("Database not configured")
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(db_path=None):
    if db_path is not None:
        configure(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        # WAL can misbehave inside OneDrive/Dropbox-synced folders (extra -wal/-shm files get locked or synced mid-write),
        # so use the classic rollback journal there. Override with SQLITE_JOURNAL_MODE=WAL|DELETE.
        import os
        mode = os.environ.get("SQLITE_JOURNAL_MODE") or ("DELETE" if any(t in str(_DB_PATH).lower() for t in ("onedrive", "dropbox")) else "WAL")
        if mode.upper() not in ("WAL", "DELETE", "TRUNCATE"): mode = "DELETE"
        c.execute(f"PRAGMA journal_mode = {mode.upper()}")
        c.executescript(SCHEMA.read_text(encoding="utf-8"))


def _j(obj):
    return json.dumps(obj, default=str) if obj is not None else None


def create_examination(patient_label=None, has_ground_truth=False, model_version=None,
                       checkpoint_sha256=None, config_snapshot=None) -> tuple:
    """Returns (id, exam_uid)."""
    uid = uuid.uuid4().hex
    with connect() as c:
        cur = c.execute(
            "INSERT INTO examinations (exam_uid, patient_label, created_at, updated_at, has_ground_truth,"
            " model_version, checkpoint_sha256, config_snapshot_json) VALUES (?,?,?,?,?,?,?,?)",
            (uid, patient_label, _now(), _now(), int(has_ground_truth), model_version, checkpoint_sha256, _j(config_snapshot)))
        return cur.lastrowid, uid


def save_uploaded_file_record(exam_id, role, original_name, stored_name, size_bytes, sha256):
    with connect() as c:
        c.execute("INSERT INTO uploaded_files (examination_id, role, original_name, stored_name, size_bytes, sha256, uploaded_at)"
                  " VALUES (?,?,?,?,?,?,?)", (exam_id, role, original_name, stored_name, size_bytes, sha256, _now()))


def update_examination_status(exam_id, status=None, stage=None, error=None, **json_fields):
    """json_fields may include validation_json, warnings_json, timings_json (dict/list values)."""
    sets, vals = ["updated_at=?"], [_now()]
    if status is not None: sets.append("status=?"); vals.append(status)
    if stage is not None: sets.append("current_stage=?"); vals.append(stage)
    if error is not None: sets.append("error_message=?"); vals.append(error)
    for k, v in json_fields.items():
        if k not in ("validation_json", "warnings_json", "timings_json", "review_status", "review_note", "model_version", "checkpoint_sha256", "config_snapshot_json"):
            raise ValueError(f"Unsupported field {k}")
        sets.append(f"{k}=?"); vals.append(v if isinstance(v, str) or v is None else _j(v))
    vals.append(exam_id)
    with connect() as c:
        c.execute(f"UPDATE examinations SET {', '.join(sets)} WHERE id=?", vals)


def save_segmentation_result(exam_id, prediction_file, result: dict):
    with connect() as c:
        c.execute("DELETE FROM segmentation_results WHERE examination_id=?", (exam_id,))
        c.execute("INSERT INTO segmentation_results (examination_id, prediction_file, device, class_mapping_json,"
                  " region_volumes_json, region_voxels_json, label_counts_json, inference_json, qc_json, created_at)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (exam_id, prediction_file, result.get("device"), _j(result.get("class_mapping")),
                   _j(result.get("region_volumes_ml")), _j(result.get("region_voxels")), _j(result.get("label_voxel_counts")),
                   _j({"stats": result.get("inference_stats"), "transform": result.get("transform"),
                       "geometry": result.get("output_geometry_check")}), _j(result.get("qc_warnings")), _now()))


def save_radiomics_result(exam_id, source, rows, settings, version, failures, timings) -> int:
    with connect() as c:
        c.execute("DELETE FROM radiomics_runs WHERE examination_id=? AND segmentation_source=?", (exam_id, source))
        cur = c.execute("INSERT INTO radiomics_runs (examination_id, segmentation_source, settings_json, pyradiomics_version,"
                        " failures_json, timings_json, created_at) VALUES (?,?,?,?,?,?,?)",
                        (exam_id, source, _j(settings), version, _j(failures), _j(timings), _now()))
        run_id = cur.lastrowid
        c.executemany("INSERT INTO radiomics_results (run_id, examination_id, region, modality, segmentation_source,"
                      " feature_class, feature_name, value) VALUES (?,?,?,?,?,?,?,?)",
                      [(run_id, exam_id, r["region"], r["modality"], source, r["feature_class"], r["feature_name"], r["value"]) for r in rows])
        return run_id


def save_evaluation_result(exam_id, metrics: dict, policy: str):
    with connect() as c:
        c.execute("DELETE FROM evaluation_results WHERE examination_id=?", (exam_id,))
        for region, m in metrics.items():
            c.execute("INSERT INTO evaluation_results (examination_id, region, dice, iou, precision, recall, specificity,"
                      " hd95_mm, pred_voxels, ref_voxels, notes, policy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (exam_id, region, m["dice"], m["iou"], m["precision"], m["recall"], m["specificity"], m["hd95_mm"],
                       m["pred_voxels"], m["ref_voxels"], m["notes"], policy, _now()))


def save_generated_report(exam_id, file_name, sha256, seconds, kind="pdf"):
    with connect() as c:
        c.execute("INSERT INTO generated_reports (examination_id, kind, file_name, sha256, generation_seconds, created_at)"
                  " VALUES (?,?,?,?,?,?)", (exam_id, kind, file_name, sha256, seconds, _now()))


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def get_examination(exam_id) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM examinations WHERE id=?", (exam_id,)).fetchone()
        if not r:
            return None
        e = dict(r)
        for k in ("validation_json", "warnings_json", "timings_json", "config_snapshot_json"):
            e[k[:-5]] = json.loads(e[k]) if e.get(k) else None
        e["files"] = _rows(c.execute("SELECT * FROM uploaded_files WHERE examination_id=?", (exam_id,)))
        seg = c.execute("SELECT * FROM segmentation_results WHERE examination_id=?", (exam_id,)).fetchone()
        if seg:
            s = dict(seg)
            for k in ("class_mapping_json", "region_volumes_json", "region_voxels_json", "label_counts_json", "inference_json", "qc_json"):
                s[k[:-5]] = json.loads(s[k]) if s.get(k) else None
            e["segmentation"] = s
        else:
            e["segmentation"] = None
        e["evaluation"] = _rows(c.execute("SELECT * FROM evaluation_results WHERE examination_id=? ORDER BY id", (exam_id,)))
        e["radiomics_runs"] = _rows(c.execute("SELECT id, segmentation_source, pyradiomics_version, created_at, settings_json,"
                                              " failures_json, timings_json FROM radiomics_runs WHERE examination_id=?", (exam_id,)))
        e["reports"] = _rows(c.execute("SELECT * FROM generated_reports WHERE examination_id=? ORDER BY id DESC", (exam_id,)))
        e["jobs"] = _rows(c.execute("SELECT * FROM processing_jobs WHERE examination_id=? ORDER BY id DESC", (exam_id,)))
        return e


def get_radiomics_rows(exam_id, source=None) -> list:
    q = "SELECT region, modality, segmentation_source, feature_class, feature_name, value FROM radiomics_results WHERE examination_id=?"
    args = [exam_id]
    if source:
        q += " AND segmentation_source=?"; args.append(source)
    with connect() as c:
        return _rows(c.execute(q + " ORDER BY region, modality, feature_class, feature_name", args))


def get_all_examinations() -> list:
    with connect() as c:
        return _rows(c.execute(
            "SELECT e.id, e.exam_uid, e.patient_label, e.created_at, e.status, e.model_version, e.has_ground_truth, e.review_status,"
            " (SELECT COUNT(*) FROM generated_reports g WHERE g.examination_id=e.id) AS n_reports"
            " FROM examinations e ORDER BY e.id DESC"))


def delete_examination(exam_id) -> str | None:
    """Delete DB rows (cascade). Returns the exam_uid so the caller can delete files."""
    with connect() as c:
        r = c.execute("SELECT exam_uid FROM examinations WHERE id=?", (exam_id,)).fetchone()
        if not r:
            return None
        c.execute("INSERT INTO audit_events (examination_id, event, detail, created_at) VALUES (?,?,?,?)",
                  (exam_id, "examination_deleted", None, _now()))
        c.execute("DELETE FROM examinations WHERE id=?", (exam_id,))
        return r["exam_uid"]


def log_audit(exam_id, event, detail=None):
    with connect() as c:
        c.execute("INSERT INTO audit_events (examination_id, event, detail, created_at) VALUES (?,?,?,?)",
                  (exam_id, event, detail, _now()))


# --- jobs ---------------------------------------------------------------------------------
def start_job(exam_id, job_type) -> int | None:
    """Create a job unless one is already queued/running for this examination (duplicate guard).
    Returns job id or None if a conflicting job exists. Atomic via BEGIN IMMEDIATE."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        busy = conn.execute("SELECT 1 FROM processing_jobs WHERE examination_id=? AND status IN ('queued','running')", (exam_id,)).fetchone()
        if busy:
            conn.execute("ROLLBACK"); return None
        cur = conn.execute("INSERT INTO processing_jobs (examination_id, job_type, status, created_at) VALUES (?,?,?,?)",
                           (exam_id, job_type, "queued", _now()))
        conn.execute("COMMIT")
        return cur.lastrowid
    finally:
        conn.close()


def update_job(job_id, status=None, stage=None, error=None, detail=None):
    sets, vals = [], []
    if status: sets.append("status=?"); vals.append(status)
    if status in ("completed", "failed", "interrupted"): sets.append("finished_at=?"); vals.append(_now())
    if stage: sets.append("stage=?"); vals.append(stage)
    if error: sets.append("error=?"); vals.append(error)
    if detail is not None: sets.append("detail_json=?"); vals.append(_j(detail))
    if not sets: return
    vals.append(job_id)
    with connect() as c:
        c.execute(f"UPDATE processing_jobs SET {', '.join(sets)} WHERE id=?", vals)


def get_latest_job(exam_id) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM processing_jobs WHERE examination_id=? ORDER BY id DESC LIMIT 1", (exam_id,)).fetchone()
        return dict(r) if r else None


def mark_interrupted_jobs() -> int:
    """Called at startup: jobs left 'queued/running' by a previous process cannot still be running."""
    with connect() as c:
        ids = [r["examination_id"] for r in c.execute("SELECT examination_id FROM processing_jobs WHERE status IN ('queued','running')")]
        c.execute("UPDATE processing_jobs SET status='interrupted', finished_at=?, error='Application restarted during processing' WHERE status IN ('queued','running')", (_now(),))
        c.execute("UPDATE examinations SET status='interrupted', updated_at=? WHERE status='processing'", (_now(),))
        return len(ids)


def get_setting(key, default=None):
    with connect() as c:
        r = c.execute("SELECT value FROM application_settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default


def set_setting(key, value):
    with connect() as c:
        c.execute("INSERT INTO application_settings (key,value,updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                  (key, str(value), _now()))


def dashboard_stats() -> dict:
    """All dashboard numbers come from the DB."""
    with connect() as c:
        q = lambda s: c.execute(s).fetchone()[0]
        return {
            "total_exams": q("SELECT COUNT(*) FROM examinations"),
            "successful_segmentations": q("SELECT COUNT(*) FROM segmentation_results"),
            "radiomics_analyses": q("SELECT COUNT(*) FROM radiomics_runs"),
            "evaluations": q("SELECT COUNT(DISTINCT examination_id) FROM evaluation_results"),
            "failed": q("SELECT COUNT(*) FROM examinations WHERE status IN ('failed','interrupted')"),
            "reports": q("SELECT COUNT(*) FROM generated_reports"),
            "completed": q("SELECT COUNT(*) FROM examinations WHERE status='segmented'"),
            "pending": q("SELECT COUNT(*) FROM examinations WHERE status IN ('uploaded','validated','processing')"),
            "radiomics_by_source": {r[0]: r[1] for r in c.execute("SELECT segmentation_source, COUNT(*) FROM radiomics_runs GROUP BY segmentation_source")},
            "wt_volumes": [{"id": r[0], "label": r[1], "wt_ml": r[2]} for r in c.execute(
                "SELECT e.id, e.patient_label, json_extract(s.region_volumes_json, '$.WT') FROM segmentation_results s JOIN examinations e ON e.id=s.examination_id ORDER BY e.id DESC LIMIT 12")],
            "status_counts": {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM examinations GROUP BY status")},
            "per_day": [dict(r) for r in c.execute("SELECT substr(created_at,1,10) AS day, COUNT(*) AS n FROM examinations GROUP BY day ORDER BY day DESC LIMIT 14")],
            "recent_events": [dict(r) for r in c.execute("SELECT examination_id, event, created_at FROM audit_events ORDER BY id DESC LIMIT 8")],
            "mean_dice": {r["region"]: r["m"] for r in c.execute("SELECT region, AVG(dice) AS m FROM evaluation_results WHERE dice IS NOT NULL GROUP BY region")},
        }


def performance_summary() -> dict:
    """Aggregate evaluation_results over all evaluated examinations (descriptive; NOT an independent test set)."""
    import statistics as st
    out = {"n_cases": 0, "regions": {}}
    with connect() as c:
        out["n_cases"] = c.execute("SELECT COUNT(DISTINCT examination_id) FROM evaluation_results").fetchone()[0]
        for region in ("WT", "TC", "ET"):
            reg = {}
            for k in ("dice", "iou", "precision", "recall", "specificity", "hd95_mm"):
                vals = [r[0] for r in c.execute(f"SELECT {k} FROM evaluation_results WHERE region=? AND {k} IS NOT NULL", (region,))]
                n_all = c.execute("SELECT COUNT(*) FROM evaluation_results WHERE region=?", (region,)).fetchone()[0]
                reg[k] = {"n": len(vals), "n_undefined": n_all - len(vals), "mean": st.fmean(vals) if vals else None, "sd": st.stdev(vals) if len(vals) > 1 else None,
                          "median": st.median(vals) if vals else None}
            out["regions"][region] = reg
        out["cases"] = [dict(r) for r in c.execute("SELECT e.id, e.patient_label, v.region, v.dice, v.iou FROM evaluation_results v JOIN examinations e ON e.id=v.examination_id ORDER BY e.id DESC, v.region LIMIT 60")]
    return out
