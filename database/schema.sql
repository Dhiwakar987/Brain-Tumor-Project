PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS examinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_uid TEXT NOT NULL UNIQUE,              -- opaque id used for folders / URLs of files
    patient_label TEXT,                         -- free text, de-identified label supplied by user
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',    -- uploaded|validated|processing|segmented|failed|interrupted
    current_stage TEXT,
    has_ground_truth INTEGER NOT NULL DEFAULT 0,
    model_version TEXT,
    checkpoint_sha256 TEXT,
    review_status TEXT NOT NULL DEFAULT 'not_reviewed',   -- not_reviewed|reviewed|requires_attention
    review_note TEXT,
    validation_json TEXT,                       -- upload-time validation result
    warnings_json TEXT,
    timings_json TEXT,
    config_snapshot_json TEXT,                  -- model/preprocessing config used (preserved per exam)
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS uploaded_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                         -- t1|t1ce|t2|flair|ground_truth
    original_name TEXT,
    stored_name TEXT NOT NULL,
    size_bytes INTEGER,
    sha256 TEXT,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segmentation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    prediction_file TEXT NOT NULL,              -- relative to outputs/<exam_uid>/
    device TEXT,
    class_mapping_json TEXT,
    region_volumes_json TEXT,
    region_voxels_json TEXT,
    label_counts_json TEXT,
    inference_json TEXT,
    qc_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS radiomics_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    segmentation_source TEXT NOT NULL,          -- predicted|ground_truth
    settings_json TEXT NOT NULL,
    pyradiomics_version TEXT,
    failures_json TEXT,
    timings_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS radiomics_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES radiomics_runs(id) ON DELETE CASCADE,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    region TEXT NOT NULL,
    modality TEXT NOT NULL,                     -- t1|t1ce|t2|flair|mask_only (shape)
    segmentation_source TEXT NOT NULL,
    feature_class TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL
);
CREATE INDEX IF NOT EXISTS idx_rad_exam ON radiomics_results(examination_id, run_id);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    region TEXT NOT NULL,
    dice REAL, iou REAL, precision REAL, recall REAL, specificity REAL, hd95_mm REAL,
    pred_voxels INTEGER, ref_voxels INTEGER,
    notes TEXT,
    policy TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generated_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'pdf',
    file_name TEXT NOT NULL,
    sha256 TEXT,
    generation_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER NOT NULL REFERENCES examinations(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,                     -- segmentation|radiomics|evaluation
    status TEXT NOT NULL,                       -- queued|running|completed|failed|interrupted
    stage TEXT,
    detail_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS application_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id INTEGER,
    event TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
