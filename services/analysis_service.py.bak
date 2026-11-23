"""Service layer: all ML / radiomics / evaluation orchestration. Flask routes only call this."""
import json
import logging
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from database import db
from model.checkpoint_validation import sha256_file
from model.preprocessing import validate_modalities
from radiomics_analysis import extractor as rx
from radiomics_analysis.feature_export import compare_radiomics_features, save_radiomics_csv, save_radiomics_excel
from segmentation import metrics as mt
from segmentation.nifti_io import NiftiError, load_nifti
from segmentation.prediction import run_segmentation
from segmentation.regions import REGION_LABELS, generate_region_masks, validate_label_values
from services import file_service as fs
from visualization import mri_viewer as mv

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)  # serialises heavy jobs (1 at a time) -> no conflicting writes


class AnalysisService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def upload_dir(self, uid): return fs.exam_dir(self.cfg.UPLOAD_DIR, uid)
    def output_dir(self, uid): return fs.exam_dir(self.cfg.OUTPUT_DIR, uid)

    def _file_path(self, exam, role):
        for f in exam["files"]:
            if f["role"] == role:
                return self.upload_dir(exam["exam_uid"]) / f["stored_name"]
        return None

    def prediction_path(self, exam):
        return self.output_dir(exam["exam_uid"]) / exam["segmentation"]["prediction_file"] if exam.get("segmentation") else None

    # ---------------------------------------------------------------- upload
    def create_examination(self, files: dict, patient_label: str | None) -> dict:
        """Validate + store an upload. files: role -> FileStorage (ground_truth optional).
        Returns {ok, id, errors, warnings}. On any failure nothing is kept on disk/DB."""
        t0 = time.perf_counter()
        errors = [f"Missing {r.upper()} file" for r in fs.MODALITIES if not files.get(r) or not files[r].filename]
        if errors:
            return {"ok": False, "errors": errors}
        mc = self.cfg.model_config()
        has_gt = bool(files.get("ground_truth") and files["ground_truth"].filename)
        ckpt_hash = sha256_file(self.cfg.CHECKPOINT_PATH) if self.cfg.CHECKPOINT_PATH.is_file() else None
        label = (patient_label or "").strip()[:60] or None
        exam_id, uid = db.create_examination(label, has_gt, f"unet-{mc['encoder_name']}", ckpt_hash, mc)
        up = self.upload_dir(uid)
        try:
            records = []
            for role in fs.MODALITIES + (("ground_truth",) if has_gt else ()):
                records.append(fs.save_upload(files[role], role, up, self.cfg.ALLOWED_EXTENSIONS))
            vols, mask = {}, None
            for r in records:
                p = up / r["stored_name"]
                if r["role"] == "ground_truth":
                    mask = load_nifti(p)
                else:
                    vols[r["role"]] = load_nifti(p)
            validation = validate_modalities(vols, mask=mask)
            if mask is not None:
                bad = validate_label_values(mask.data.astype(np.int16))
                if bad:
                    validation["errors"].append(f"Ground-truth contains unexpected label values {bad}; expected BraTS labels 0,1,2,4")
                    validation["ok"] = False
            if not validation["ok"]:
                raise fs.UploadError("; ".join(validation["errors"]))
            for r in records:
                db.save_uploaded_file_record(exam_id, r["role"], r["original_name"], r["stored_name"], r["size_bytes"], r["sha256"])
            db.update_examination_status(exam_id, status="validated", stage="validated", validation_json=validation,
                                         warnings_json=validation["warnings"], timings_json={"upload_validation_s": round(time.perf_counter() - t0, 3)})
            db.log_audit(exam_id, "upload_validated")
            return {"ok": True, "id": exam_id, "warnings": validation["warnings"]}
        except (fs.UploadError, NiftiError) as exc:
            self._discard(exam_id, uid)
            return {"ok": False, "errors": [str(exc)]}
        except Exception as exc:  # unexpected
            log.exception("Upload failed")
            self._discard(exam_id, uid)
            return {"ok": False, "errors": [f"Unexpected error while processing upload: {type(exc).__name__}"]}

    def _discard(self, exam_id, uid):
        fs.remove_tree(self.upload_dir(uid)); db.delete_examination(exam_id)

    # ---------------------------------------------------------------- jobs
    def _submit(self, exam_id, job_type, fn, *args):
        job_id = db.start_job(exam_id, job_type)
        if job_id is None:
            return {"ok": False, "error": "A job is already queued or running for this examination."}

        def runner():
            db.update_job(job_id, status="running")
            try:
                detail = fn(job_id, *args)
                db.update_job(job_id, status="completed", stage="done", detail=detail)
            except Exception as exc:
                log.exception("Job %s failed", job_type)
                msg = f"{type(exc).__name__}: {exc}"
                db.update_job(job_id, status="failed", error=msg)
                if job_type == "segmentation":
                    db.update_examination_status(exam_id, status="failed", error=msg)
        _executor.submit(runner)
        return {"ok": True, "job_id": job_id}

    def _stage(self, job_id, exam_id=None):
        def cb(stage):
            db.update_job(job_id, stage=stage)
            if exam_id: db.update_examination_status(exam_id, stage=stage)
        return cb

    def start_segmentation(self, exam_id, wait=False):
        exam = db.get_examination(exam_id)
        if not exam: return {"ok": False, "error": "Examination not found"}
        if exam["status"] == "uploaded":
            return {"ok": False, "error": "Examination has not passed validation"}
        res = self._submit(exam_id, "segmentation", self._segment_job, exam_id)
        if wait and res["ok"]:
            _executor.submit(lambda: None).result()
        return res

    def _segment_job(self, job_id, exam_id):
        exam = db.get_examination(exam_id)
        db.update_examination_status(exam_id, status="processing", stage="loading_inputs", error="")
        t0 = time.perf_counter()
        mc = self.cfg.model_config()
        vols = {r: load_nifti(self._file_path(exam, r)) for r in fs.MODALITIES}
        load_s = round(time.perf_counter() - t0, 3)
        out_dir = self.output_dir(exam["exam_uid"]); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "prediction.nii.gz"
        res = run_segmentation(vols, self._file_path(exam, "t1"), out, mc, self.cfg.CHECKPOINT_PATH,
                               self.cfg.DEVICE, self.cfg.INFERENCE_BATCH_SIZE, self.cfg.ALLOW_UNSAFE_CHECKPOINT_PICKLE,
                               stage_cb=self._stage(job_id, exam_id))
        res["timings"]["input_loading_s"] = load_s
        timings = {**(exam.get("timings") or {}), **res["timings"], "segmentation_total_s": round(time.perf_counter() - t0, 3)}
        db.save_segmentation_result(exam_id, out.name, res)
        warns = list(exam.get("warnings") or []) + res["qc_warnings"]
        db.update_examination_status(exam_id, status="segmented", stage="done", warnings_json=warns, timings_json=timings,
                                     config_snapshot_json=mc, checkpoint_sha256=sha256_file(self.cfg.CHECKPOINT_PATH))
        db.log_audit(exam_id, "segmentation_completed")
        return {"elapsed_s": timings["segmentation_total_s"]}

    def start_radiomics(self, exam_id, source="predicted", regions=None, modalities=None):
        exam = db.get_examination(exam_id)
        if not exam: return {"ok": False, "error": "Examination not found"}
        if source == "predicted" and not exam["segmentation"]:
            return {"ok": False, "error": "Run segmentation first"}
        if source == "ground_truth" and not exam["has_ground_truth"]:
            return {"ok": False, "error": "No ground-truth mask was uploaded"}
        regions = [r for r in (regions or rx.REGIONS) if r in rx.REGIONS] or list(rx.REGIONS)
        modalities = [m for m in (modalities or fs.MODALITIES) if m in fs.MODALITIES] or list(fs.MODALITIES)
        return self._submit(exam_id, "radiomics", self._radiomics_job, exam_id, source, regions, modalities)

    def _label_volume(self, exam, source):
        p = self.prediction_path(exam) if source == "predicted" else self._file_path(exam, "ground_truth")
        return load_nifti(p, dtype=np.float32)

    def _radiomics_job(self, job_id, exam_id, source, regions, modalities):
        exam = db.get_examination(exam_id)
        t0 = time.perf_counter()
        db.update_job(job_id, stage="loading")
        vols = {m: load_nifti(self._file_path(exam, m)) for m in modalities}
        lab = self._label_volume(exam, source).data.astype(np.int16)
        masks = generate_region_masks(lab)
        db.update_job(job_id, stage="extracting_features")
        settings = self.cfg.radiomics_settings()
        out = rx.extract_all_radiomics_features(vols, masks, settings, source, regions, modalities,
                                                progress_cb=lambda r, m: db.update_job(job_id, stage=f"extracted {r}/{m}"))
        if not out["rows"] and out["failures"]:
            raise RuntimeError("All radiomics extractions failed: " + out["failures"][0]["error"])
        db.save_radiomics_result(exam_id, source, out["rows"], {**settings, "regions": regions, "modalities": modalities},
                                 out["pyradiomics_version"], out["failures"], out["timings"])
        t = dict(exam.get("timings") or {}); t[f"radiomics_{source}_s"] = round(time.perf_counter() - t0, 3)
        t[f"radiomics_{source}_by_region_modality_s"] = out["timings"]
        db.update_examination_status(exam_id, timings_json=t)
        db.log_audit(exam_id, f"radiomics_completed_{source}")
        return {"n_features": len(out["rows"]), "n_failures": len(out["failures"])}

    def start_evaluation(self, exam_id):
        exam = db.get_examination(exam_id)
        if not exam: return {"ok": False, "error": "Examination not found"}
        if not exam["has_ground_truth"]:
            return {"ok": False, "error": "No ground-truth mask supplied: quantitative accuracy cannot be calculated."}
        if not exam["segmentation"]:
            return {"ok": False, "error": "Run segmentation first"}
        return self._submit(exam_id, "evaluation", self._evaluation_job, exam_id)

    def _evaluation_job(self, job_id, exam_id):
        exam = db.get_examination(exam_id)
        t0 = time.perf_counter()
        pred = self._label_volume(exam, "predicted"); ref = self._label_volume(exam, "ground_truth")
        t1 = load_nifti(self._file_path(exam, "t1"))
        if pred.shape != ref.shape or not np.allclose(pred.affine, ref.affine, atol=1e-3):
            raise ValueError("Prediction and reference geometry differ; evaluation refused")
        res = mt.evaluate_regions(pred.data.astype(np.int16), ref.data.astype(np.int16), t1.zooms)
        db.save_evaluation_result(exam_id, res, mt.EMPTY_MASK_POLICY)
        t = dict(exam.get("timings") or {}); t["evaluation_s"] = round(time.perf_counter() - t0, 3)
        db.update_examination_status(exam_id, timings_json=t)
        db.log_audit(exam_id, "evaluation_completed")
        return {"regions": list(res)}

    # ---------------------------------------------------------------- viewer
    @staticmethod
    @lru_cache(maxsize=6)
    def _cached(path_str, mtime):
        return load_nifti(path_str)

    def _vol(self, path):
        return self._cached(str(path), Path(path).stat().st_mtime_ns)

    def slice_payload(self, exam, modality, view, index, region, source):
        if view not in mv.VIEWS or modality not in fs.MODALITIES or region not in ("ALL", "WT", "TC", "ET"):
            raise ValueError("Invalid viewer parameters")
        img = self._vol(self._file_path(exam, modality))
        n = mv.n_slices(img.shape, view)
        if index is None:
            index = (mv.best_slice_index(self._vol(self.prediction_path(exam)).data, view) if exam["segmentation"] else n // 2)
        index = int(np.clip(index, 0, n - 1))
        win = mv.display_window(img.data)
        base = mv.display_scale(mv.extract_slice(img.data, view, index), win)
        xl, yl = mv.axis_labels(img.axcodes, view)
        payload = {"xlabel": xl, "ylabel": yl, "orientation": "".join(img.axcodes), "n_slices": n, "index": index, "aspect": mv.inplane_aspect(img.zooms, view), "image": base.tolist(), "mask": None, "region": region}
        lp = None
        if source == "predicted" and exam["segmentation"]: lp = self.prediction_path(exam)
        elif source == "ground_truth" and exam["has_ground_truth"]: lp = self._file_path(exam, "ground_truth")
        if lp is not None:
            lab = self._vol(lp).data.astype(np.int16)
            m = mv.extract_slice(mv.region_display_mask(lab, region), view, index).astype(float)
            payload["mask"] = np.where(m > 0, m, np.nan).tolist()
            payload["mask"] = [[None if (v != v) else v for v in row] for row in payload["mask"]]
        return payload

    # ---------------------------------------------------------------- measurements / compare / 3-D
    def measurements(self, exam) -> dict | None:
        """Physical measurements of predicted (and GT, if given) regions; cached in outputs/<uid>/measurements.json."""
        if not exam.get("segmentation"):
            return None
        cache = self.output_dir(exam["exam_uid"]) / "measurements.json"
        pred_path = self.prediction_path(exam)
        if cache.is_file() and cache.stat().st_mtime >= pred_path.stat().st_mtime:
            return json.loads(cache.read_text(encoding="utf-8"))
        zooms = self._vol(self._file_path(exam, "t1")).zooms
        from segmentation.measurements import all_region_measurements
        res = {"predicted": all_region_measurements(self._vol(pred_path).data.astype(np.int16), zooms), "ground_truth": None,
               "method": "voxel counts x header voxel spacing; surface via marching cubes (scikit-image)"}
        if exam["has_ground_truth"]:
            res["ground_truth"] = all_region_measurements(self._vol(self._file_path(exam, "ground_truth")).data.astype(np.int16), zooms)
        cache.write_text(json.dumps(res), encoding="utf-8")
        return res

    def compare_payload(self, exam, modality, view, index, region):
        """Same slice for the three panels: MRI, ground truth (or None), AI prediction."""
        if view not in mv.VIEWS or modality not in fs.MODALITIES or region not in ("ALL", "WT", "TC", "ET") or not exam["segmentation"]:
            raise ValueError("Invalid parameters or no segmentation yet")
        img = self._vol(self._file_path(exam, modality)); n = mv.n_slices(img.shape, view)
        pred = self._vol(self.prediction_path(exam)).data.astype(np.int16)
        if index is None: index = mv.best_slice_index(pred, view)
        index = int(np.clip(index, 0, n - 1))
        def msk(lab):
            m = mv.extract_slice(mv.region_display_mask(lab, region), view, index).astype(float)
            return [[None if v == 0 else v for v in row] for row in m.tolist()]
        xl, yl = mv.axis_labels(img.axcodes, view)
        gt = None
        if exam["has_ground_truth"]:
            gt = msk(self._vol(self._file_path(exam, "ground_truth")).data.astype(np.int16))
        return {"n_slices": n, "index": index, "aspect": mv.inplane_aspect(img.zooms, view), "xlabel": xl, "ylabel": yl, "orientation": "".join(img.axcodes),
                "image": mv.display_scale(mv.extract_slice(img.data, view, index), mv.display_window(img.data)).tolist(),
                "prediction": msk(pred), "ground_truth": gt, "region": region}

    def mesh_payload(self, exam, modality="flair"):
        if not exam["segmentation"] or modality not in fs.MODALITIES:
            raise ValueError("Segmentation required")
        cache = self.output_dir(exam["exam_uid"]) / f"mesh3d_{modality}.json"
        pred_path = self.prediction_path(exam)
        if cache.is_file() and cache.stat().st_mtime >= pred_path.stat().st_mtime:
            return json.loads(cache.read_text(encoding="utf-8"))
        from visualization.mesh3d import build_meshes
        img = self._vol(self._file_path(exam, modality))
        m = build_meshes(img.data, self._vol(pred_path).data.astype(np.int16), img.zooms)
        cache.write_text(json.dumps(m), encoding="utf-8")
        return m

    # ---------------------------------------------------------------- full export (CSV / XLSX / JSON)
    def export_tables(self, exam_id) -> dict:
        import pandas as pd
        exam = db.get_examination(exam_id)
        cfg = exam["config_snapshot"] or {}
        seg = exam["segmentation"]
        summary = [("case_label", exam["patient_label"] or ""), ("analysis_date_utc", exam["created_at"]), ("status", exam["status"]),
                   ("modalities_used", ", ".join(f["role"] for f in exam["files"] if f["role"] != "ground_truth")),
                   ("ground_truth_provided", "yes" if exam["has_ground_truth"] else "no"),
                   ("model", f"U-Net / {cfg.get('encoder_name')} ({cfg.get('in_channels')} in, {cfg.get('classes')} classes)"),
                   ("checkpoint_sha256", exam["checkpoint_sha256"] or ""), ("device", seg["device"] if seg else ""),
                   ("normalization", (cfg.get("preprocessing") or {}).get("normalization", "")),
                   ("preprocessing_verified_against_training", str((cfg.get("verification") or {}).get("preprocessing_verified_against_training"))),
                   ("classification", "not available (no trained classifier in this project)"),
                   ("explainability", "not available (no Grad-CAM++/TreeSHAP implemented)"),
                   ("disclaimer", "Research prototype; not clinically validated")]
        meas = []
        m = self.measurements(exam)
        if m:
            for source in ("predicted", "ground_truth"):
                for region, d in (m.get(source) or {}).items():
                    ext = d["bbox_extent_mm"] or [None] * 3
                    meas.append({"source": source, "region": region, "voxel_count": d["voxel_count"], "volume_mm3": d["volume_mm3"], "volume_ml": d["volume_ml"],
                                 "bbox_extent_axis0_mm": ext[0], "bbox_extent_axis1_mm": ext[1], "bbox_extent_axis2_mm": ext[2],
                                 "surface_area_mm2": d["surface_area_mm2"], "sphericity": d["sphericity"], "max_3d_diameter_mm": d["max_3d_diameter_mm"],
                                 "surface_to_volume_per_mm": d["surface_to_volume_ratio_per_mm"], "spacing_mm": "x".join(f"{v:g}" for v in d["spacing_mm"])})
        rad = db.get_radiomics_rows(exam_id)
        ev = [{k: e[k] for k in ("region", "dice", "iou", "precision", "recall", "specificity", "hd95_mm", "pred_voxels", "ref_voxels", "notes")} for e in exam["evaluation"]]
        return {"summary": pd.DataFrame(summary, columns=["item", "value"]), "measurements": pd.DataFrame(meas), "radiomics": pd.DataFrame(rad),
                "evaluation": pd.DataFrame(ev)}

    def export_file(self, exam_id, fmt) -> Path:
        import pandas as pd
        exam = db.get_examination(exam_id); od = self.output_dir(exam["exam_uid"]); od.mkdir(parents=True, exist_ok=True)
        t = self.export_tables(exam_id)
        if fmt == "xlsx":
            p = od / "analysis_export.xlsx"
            with pd.ExcelWriter(p, engine="openpyxl") as xw:
                for name, df in t.items():
                    (df if len(df) else pd.DataFrame({"info": ["no data (not run / not available)"]})).to_excel(xw, sheet_name=name, index=False)
            return p
        if fmt == "csv":  # long format: one file, section column
            parts = []
            for name, df in t.items():
                if not len(df): continue
                if name == "summary": parts.append(df.rename(columns={"item": "name"}).assign(section=name, region="", modality=""))
                elif name == "radiomics": parts.append(df.rename(columns={"feature_name": "name"}).assign(section="radiomics:" + df["feature_class"], name=df["feature_name"])[["section", "region", "modality", "name", "value"]])
                else:
                    long = df.melt(id_vars=[c for c in ("source", "region") if c in df.columns], var_name="name", value_name="value")
                    parts.append(long.assign(section=name, modality=""))
            out = pd.concat(parts, ignore_index=True)
            p = od / "analysis_export.csv"; out.to_csv(p, index=False); return p
        raise ValueError("format must be csv or xlsx")

    def export_json(self, exam_id) -> dict:
        t = self.export_tables(exam_id)
        return {"summary": dict(zip(t["summary"]["item"], t["summary"]["value"])), "measurements": t["measurements"].to_dict("records"),
                "radiomics": t["radiomics"].to_dict("records"), "radiomics_settings": [json.loads(r["settings_json"]) for r in db.get_examination(exam_id)["radiomics_runs"]],
                "evaluation": t["evaluation"].to_dict("records"), "reproducibility": self.metadata_package(exam_id)}

    # ---------------------------------------------------------------- environment / metadata
    def environment_info(self) -> dict:
        import torch, nibabel, scipy
        info = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
                "nibabel": nibabel.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "device_setting": self.cfg.DEVICE, "cuda_available": torch.cuda.is_available()}
        try:
            import segmentation_models_pytorch as smp; info["segmentation_models_pytorch"] = smp.__version__
        except Exception: pass
        try: info["pyradiomics"] = rx.pyradiomics_version()
        except Exception: info["pyradiomics"] = "not installed"
        return info

    def metadata_package(self, exam_id) -> dict:
        from config import APP_VERSION
        exam = db.get_examination(exam_id)
        runs = []
        for r in exam["radiomics_runs"]:
            runs.append({"source": r["segmentation_source"], "pyradiomics_version": r["pyradiomics_version"], "created_at": r["created_at"],
                         "settings": json.loads(r["settings_json"]), "failures": json.loads(r["failures_json"] or "[]")})
        seg = exam["segmentation"]
        return {
            "examination_id": exam["id"], "created_at": exam["created_at"], "status": exam["status"],
            "application_version": APP_VERSION, "environment": self.environment_info(),
            "inputs": [{"role": f["role"], "original_name": f["original_name"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in exam["files"]],
            "checkpoint_sha256": exam["checkpoint_sha256"], "model_config_used": exam["config_snapshot"],
            "input_validation": exam["validation"], "segmentation": (None if not seg else {
                "device": seg["device"], "class_mapping": seg["class_mapping"], "region_volumes_ml": seg["region_volumes"],
                "region_voxels": seg["region_voxels"], "label_voxel_counts": seg["label_counts"], "inference": seg["inference"], "qc_warnings": seg["qc"],
                "output": "prediction.nii.gz"}),
            "radiomics_runs": runs,
            "evaluation": [{k: e[k] for k in ("region", "dice", "iou", "precision", "recall", "specificity", "hd95_mm", "pred_voxels", "ref_voxels", "notes")} for e in exam["evaluation"]],
            "evaluation_policy": mt.EMPTY_MASK_POLICY if exam["evaluation"] else None,
            "timings_s": exam["timings"], "warnings": exam["warnings"], "review_status": exam["review_status"],
            "jobs": [{"type": j["job_type"], "status": j["status"], "stage": j["stage"], "error": j["error"], "created_at": j["created_at"], "finished_at": j["finished_at"]} for j in exam["jobs"]],
            "note": "Research prototype metadata. Paths are intentionally omitted.",
        }

    # ---------------------------------------------------------------- exports / report
    def radiomics_csv(self, exam_id, source=None, excel=False) -> Path:
        rows = db.get_radiomics_rows(exam_id, source)
        exam = db.get_examination(exam_id)
        out = self.output_dir(exam["exam_uid"]) / f"radiomics_{source or 'all'}.{'xlsx' if excel else 'csv'}"
        return (save_radiomics_excel(rows, out, json.loads(exam["radiomics_runs"][0]["settings_json"]) if exam["radiomics_runs"] else None)
                if excel else save_radiomics_csv(rows, out))

    def radiomics_comparison(self, exam_id):
        pred = db.get_radiomics_rows(exam_id, "predicted"); ref = db.get_radiomics_rows(exam_id, "ground_truth")
        if not pred or not ref: return None
        return compare_radiomics_features(pred, ref)

    def generate_report(self, exam_id) -> Path:
        from config import APP_VERSION
        from reports.pdf_generator import generate_pdf_report
        from visualization.plots import metrics_bar, snapshot_figure
        t0 = time.perf_counter()
        exam = db.get_examination(exam_id)
        od = self.output_dir(exam["exam_uid"]); od.mkdir(parents=True, exist_ok=True)
        snap = chart = None
        if exam["segmentation"]:
            snap = snapshot_figure(self._vol(self._file_path(exam, "flair")).data, self._vol(self.prediction_path(exam)).data.astype(np.int16),
                                   self._vol(self._file_path(exam, "flair")).zooms, od / "snapshot.png", "FLAIR")
        three = None
        if exam["segmentation"]:
            from visualization.plots import three_panel_figure
            lab_gt = self._vol(self._file_path(exam, "ground_truth")).data.astype(np.int16) if exam["has_ground_truth"] else None
            three = three_panel_figure(self._vol(self._file_path(exam, "flair")).data, self._vol(self.prediction_path(exam)).data.astype(np.int16), lab_gt,
                                       self._vol(self._file_path(exam, "flair")).zooms, od / "three_panel.png")
        if exam["evaluation"]:
            chart = metrics_bar({e["region"]: e for e in exam["evaluation"]}, od / "dice.png")
        name = f"report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        generate_pdf_report(od / name, exam, db.get_radiomics_rows(exam_id), self.environment_info(), snap, chart, APP_VERSION,
                           measurements=self.measurements(exam), three_panel_path=three)
        secs = round(time.perf_counter() - t0, 3)
        db.save_generated_report(exam_id, name, fs.sha256_of(od / name), secs)
        t = dict(exam.get("timings") or {}); t["report_generation_s"] = secs
        db.update_examination_status(exam_id, timings_json=t)
        return od / name

    # ---------------------------------------------------------------- delete / consistency
    def delete_examination(self, exam_id) -> dict:
        exam = db.get_examination(exam_id)
        if not exam: return {"ok": False, "error": "Not found"}
        if exam["status"] == "processing":
            return {"ok": False, "error": "Examination is being processed; try again when it finishes."}
        uid = db.delete_examination(exam_id)
        ok_up = fs.remove_tree(self.upload_dir(uid)); ok_out = fs.remove_tree(self.output_dir(uid))
        if not (ok_up and ok_out):
            log.error("Partial file deletion for exam %s", exam_id)
            return {"ok": True, "warning": "Database records were deleted but some files could not be removed; see logs."}
        return {"ok": True}

    def consistency_check(self) -> dict:
        """Detect DB rows whose files are missing and on-disk folders without DB rows."""
        exams = db.get_all_examinations()
        uids = {e["exam_uid"] for e in exams}
        missing = []
        for e in exams:
            full = db.get_examination(e["id"])
            for f in full["files"]:
                if not (self.upload_dir(e["exam_uid"]) / f["stored_name"]).is_file():
                    missing.append({"exam": e["id"], "file": f["stored_name"]})
            if full["segmentation"] and not self.prediction_path(full).is_file():
                missing.append({"exam": e["id"], "file": "prediction"})
        orphans = [p.name for base in (self.cfg.UPLOAD_DIR, self.cfg.OUTPUT_DIR) if base.exists() for p in base.iterdir()
                   if p.is_dir() and p.name not in uids]
        return {"missing_files": missing, "orphan_folders": sorted(set(orphans)), "ok": not missing and not orphans}
