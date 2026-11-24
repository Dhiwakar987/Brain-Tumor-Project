"""Flask application factory + routes. Heavy logic lives in services/."""

import os
import json
import logging
import io
import base64

from pathlib import Path

import numpy as np
from skimage.measure import regionprops, label
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from werkzeug.exceptions import RequestEntityTooLarge

from config import APP_VERSION, Config
from database import db
from model.model_loader import get_model_information, load_model
from model.inference import predict_slice
from model.explainability import make_segmentation_gradcam
from services import file_service as fs
from services.analysis_service import AnalysisService

csrf = CSRFProtect()


def _list_2d_demo_samples(dataset_root):
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    samples = {}

    if not images_dir.is_dir() or not masks_dir.is_dir():
        return samples

    for image_path in sorted(images_dir.glob("*.npy")):
        patient_id, separator, slice_number = image_path.stem.rpartition("_slice")
        if not separator or not slice_number.isdigit():
            continue
        if not (masks_dir / image_path.name).is_file():
            continue

        samples.setdefault(patient_id, {})[int(slice_number)] = image_path.name

    return {
        patient_id: dict(sorted(patient_samples.items()))
        for patient_id, patient_samples in sorted(samples.items())
    }


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    logging.basicConfig(
        level=getattr(logging, config_class.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("app")

    if getattr(config_class, "SECRET_KEY_WAS_GENERATED", False):
        log.warning(
            "SECRET_KEY not set: using a random per-process key. Set SECRET_KEY in .env."
        )

    csrf.init_app(app)
    db.initialize_database(config_class.DATABASE_PATH)

    n_int = db.mark_interrupted_jobs()
    if n_int:
        log.warning("%d job(s) were interrupted by a restart", n_int)

    fs.cleanup_partial_outputs(config_class.OUTPUT_DIR)
    svc = AnalysisService(config_class)
    app.extensions["analysis_service"] = svc

    def get_exam_or_404(exam_id):
        exam = db.get_examination(exam_id)
        if not exam:
            abort(404)
        return exam

    @app.context_processor
    def inject():
        return {
            "app_version": APP_VERSION,
            "max_upload_mb": config_class.MAX_UPLOAD_MB,
        }

    # ---------------- pages ----------------

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/dashboard")
    def dashboard():
        stats = db.dashboard_stats()
        recent = db.get_all_examinations()[:8]

        try:
            minfo = get_model_information(
                config_class.CHECKPOINT_PATH,
                config_class.model_config(),
                config_class.DEVICE,
            )
        except Exception as exc:
            minfo = {"error": str(exc)}

        return render_template(
            "dashboard.html",
            stats=stats,
            recent=recent,
            minfo=minfo,
        )

    @app.get("/analysis/new")
    def new_analysis():
        return render_template(
            "upload.html",
            extensions=config_class.ALLOWED_EXTENSIONS,
        )

    @app.post("/analysis/upload")
    def upload():
        files = {r: request.files.get(r) for r in fs.ROLES}
        res = svc.create_examination(
            files,
            request.form.get("patient_label"),
        )

        if request.headers.get("X-Requested-With") == "fetch":
            if res["ok"]:
                return jsonify(
                    ok=True,
                    redirect=url_for(
                        "results",
                        exam_id=res["id"],
                        autostart=1,
                    ),
                    warnings=res.get("warnings", []),
                )
            return jsonify(ok=False, errors=res["errors"]), 400

        if res["ok"]:
            flash("Upload validated.", "success")
            return redirect(
                url_for(
                    "results",
                    exam_id=res["id"],
                    autostart=1,
                )
            )

        for e in res["errors"]:
            flash(e, "danger")

        return redirect(url_for("new_analysis"))

    @app.post("/analysis/<int:exam_id>/segment")
    def segment(exam_id):
        get_exam_or_404(exam_id)
        res = svc.start_segmentation(exam_id)
        return jsonify(res), (202 if res["ok"] else 409)

    @app.get("/analysis/<int:exam_id>/status")
    def status(exam_id):
        exam = get_exam_or_404(exam_id)
        job = db.get_latest_job(exam_id)

        return jsonify(
            exam_status=exam["status"],
            error=exam["error_message"],
            job=(
                None
                if not job
                else {
                    "type": job["job_type"],
                    "status": job["status"],
                    "stage": job["stage"],
                    "error": job["error"],
                }
            ),
        )

    @app.get("/analysis/<int:exam_id>/results")
    def results(exam_id):
        exam = get_exam_or_404(exam_id)
        rad = {}

        for r in db.get_radiomics_rows(exam_id, "predicted"):
            if (
                r["feature_class"] == "shape"
                and r["region"] == "WT"
                and r["feature_name"]
                in (
                    "MeshVolume",
                    "VoxelVolume",
                    "SurfaceArea",
                    "Sphericity",
                    "Maximum3DDiameter",
                    "SurfaceVolumeRatio",
                )
            ):
                rad[r["feature_name"]] = r["value"]

        return render_template(
            "results.html",
            exam=exam,
            regions=fs.MODALITIES,
            rad_summary=rad,
            auto_start=(request.args.get("autostart") == "1"),
            auto_open_3d=(request.args.get("view3d") == "1"),
        )

    @app.get("/viewer/2d")
    @app.get("/viewer/2d/<int:exam_id>")
    def viewer_2d(exam_id=None):
        if exam_id is None:
            exams = db.get_all_examinations()
            exam = max(exams, key=lambda e: e["id"], default=None)
            if exam is None:
                flash("Upload an MRI analysis first to open the 2D viewer.", "warning")
                return redirect(url_for("history"))
            exam_id = exam["id"]
        else:
            get_exam_or_404(exam_id)

        return redirect(url_for("results", exam_id=exam_id))


    @app.get("/viewer/3d")
    @app.get("/viewer/3d/<int:exam_id>")
    def viewer_3d(exam_id=None):
        if exam_id is None:
            exams = db.get_all_examinations()
            exam = max(exams, key=lambda e: e["id"], default=None)
            if exam is None:
                flash("Upload an MRI analysis first to open the 3D viewer.", "warning")
                return redirect(url_for("history"))
            exam_id = exam["id"]
        else:
            get_exam_or_404(exam_id)

        return redirect(url_for("results", exam_id=exam_id, view3d=1))

    @app.get("/analysis/<int:exam_id>/slice")
    def slice_data(exam_id):
        exam = get_exam_or_404(exam_id)

        try:
            idx = request.args.get("index", type=int)
            payload = svc.slice_payload(
                exam,
                request.args.get("modality", "flair"),
                request.args.get("view", "axial"),
                idx,
                request.args.get("region", "ALL"),
                request.args.get("source", "predicted"),
            )
        except (ValueError, KeyError) as exc:
            return jsonify(error=str(exc)), 400

        return jsonify(payload)

    @app.get("/analysis/<int:exam_id>/compare-slice")
    def compare_slice(exam_id):
        exam = get_exam_or_404(exam_id)

        try:
            return jsonify(
                svc.compare_payload(
                    exam,
                    request.args.get("modality", "flair"),
                    request.args.get("view", "axial"),
                    request.args.get("index", type=int),
                    request.args.get("region", "ALL"),
                )
            )
        except (ValueError, KeyError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/analysis/<int:exam_id>/mesh3d")
    def mesh3d(exam_id):
        exam = get_exam_or_404(exam_id)

        try:
            return jsonify(
                svc.mesh_payload(
                    exam,
                    request.args.get("modality", "flair"),
                )
            )
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:
            log.exception("3D mesh failed")
            return (
                jsonify(
                    error=f"3D mesh could not be built: {type(exc).__name__}"
                ),
                500,
            )

    @app.get("/analysis/<int:exam_id>/measurements")
    def measurements(exam_id):
        exam = get_exam_or_404(exam_id)
        m = svc.measurements(exam)

        if m is None:
            return jsonify(error="No segmentation yet"), 404

        return jsonify(m)

    @app.get("/analysis/<int:exam_id>/export")
    def export(exam_id):
        get_exam_or_404(exam_id)
        fmt = request.args.get("format", "json")

        if fmt == "json":
            resp = app.response_class(
                json.dumps(
                    svc.export_json(exam_id),
                    indent=2,
                    default=str,
                ),
                mimetype="application/json",
            )
            resp.headers["Content-Disposition"] = (
                f"attachment; filename=exam{exam_id}_analysis.json"
            )
            return resp

        if fmt not in ("csv", "xlsx"):
            abort(400)

        return send_file(
            svc.export_file(exam_id, fmt),
            as_attachment=True,
            download_name=f"exam{exam_id}_analysis.{fmt}",
        )

    @app.post("/analysis/<int:exam_id>/review")
    def review(exam_id):
        get_exam_or_404(exam_id)
        st = request.form.get("review_status")

        if st not in (
            "not_reviewed",
            "reviewed",
            "requires_attention",
        ):
            abort(400)

        db.update_examination_status(
            exam_id,
            review_status=st,
            review_note=(request.form.get("review_note") or "")[:500],
        )
        db.log_audit(exam_id, f"review_{st}")

        flash(
            "Review status saved (this is a research-workflow flag, not clinical approval).",
            "success",
        )
        return redirect(url_for("results", exam_id=exam_id))

    @app.route(
        "/analysis/<int:exam_id>/radiomics",
        methods=["GET", "POST"],
    )
    def radiomics(exam_id):
        exam = get_exam_or_404(exam_id)

        if request.method == "POST":
            res = svc.start_radiomics(
                exam_id,
                request.form.get("source", "predicted"),
                request.form.getlist("regions"),
                request.form.getlist("modalities"),
            )

            if request.headers.get("X-Requested-With") == "fetch":
                return jsonify(res), (202 if res["ok"] else 409)

            flash(
                "Radiomics started." if res["ok"] else res["error"],
                "info" if res["ok"] else "danger",
            )
            return redirect(url_for("radiomics", exam_id=exam_id))

        source = request.args.get("source", "predicted")
        rows = db.get_radiomics_rows(exam_id, source)
        comparison = svc.radiomics_comparison(exam_id)

        return render_template(
            "radiomics.html",
            exam=exam,
            rows=rows,
            source=source,
            families=[
                "shape",
                "firstorder",
                "glcm",
                "glrlm",
                "glszm",
                "gldm",
                "ngtdm",
            ],
            comparison=(
                None
                if comparison is None
                else comparison.head(500).round(5).fillna("").to_dict("records")
            ),
            n_comparison=0 if comparison is None else len(comparison),
        )

    @app.get("/analysis/<int:exam_id>/radiomics/download")
    def radiomics_download(exam_id):
        exam = get_exam_or_404(exam_id)
        source = request.args.get("source", "predicted")

        if source not in ("predicted", "ground_truth"):
            abort(400)

        if not db.get_radiomics_rows(exam_id, source):
            flash("No radiomics results to export.", "warning")
            return redirect(url_for("radiomics", exam_id=exam_id))

        excel = request.args.get("format") == "xlsx"

        return send_file(
            svc.radiomics_csv(exam_id, source, excel),
            as_attachment=True,
            download_name=(
                f"radiomics_exam{exam_id}_{source}."
                f"{'xlsx' if excel else 'csv'}"
            ),
        )

    @app.get("/analysis/<int:exam_id>/radiomics/data")
    def radiomics_data(exam_id):
        get_exam_or_404(exam_id)
        return jsonify(
            db.get_radiomics_rows(
                exam_id,
                request.args.get("source", "predicted"),
            )
        )

    @app.route(
        "/analysis/<int:exam_id>/evaluate",
        methods=["GET", "POST"],
    )
    def evaluate(exam_id):
        exam = get_exam_or_404(exam_id)

        if request.method == "POST":
            res = svc.start_evaluation(exam_id)

            if request.headers.get("X-Requested-With") == "fetch":
                return jsonify(res), (202 if res["ok"] else 409)

            flash(
                "Evaluation started." if res["ok"] else res["error"],
                "info" if res["ok"] else "danger",
            )
            return redirect(url_for("evaluate", exam_id=exam_id))

        from segmentation.metrics import EMPTY_MASK_POLICY

        return render_template(
            "evaluation.html",
            exam=exam,
            policy=EMPTY_MASK_POLICY,
        )

    @app.get("/analysis/<int:exam_id>/report")
    def report(exam_id):
        exam = get_exam_or_404(exam_id)
        return render_template("report.html", exam=exam)

    @app.post("/analysis/<int:exam_id>/report/generate")
    def report_generate(exam_id):
        get_exam_or_404(exam_id)

        try:
            svc.generate_report(exam_id)
            flash("Report generated.", "success")
        except Exception as exc:
            log.exception("Report failed")
            flash(
                f"Report generation failed: {type(exc).__name__}",
                "danger",
            )

        return redirect(url_for("report", exam_id=exam_id))

    @app.get("/analysis/<int:exam_id>/download-report")
    def download_report(exam_id):
        exam = get_exam_or_404(exam_id)

        if not exam["reports"]:
            flash("Generate a report first.", "warning")
            return redirect(url_for("report", exam_id=exam_id))

        name = exam["reports"][0]["file_name"]
        path = (svc.output_dir(exam["exam_uid"]) / name).resolve()

        if (
            svc.output_dir(exam["exam_uid"]).resolve() not in path.parents
            or not path.is_file()
        ):
            abort(404)

        return send_file(
            path,
            as_attachment=True,
            download_name=f"exam{exam_id}_report.pdf",
            mimetype="application/pdf",
        )

    @app.get("/analysis/<int:exam_id>/download/prediction")
    def download_prediction(exam_id):
        exam = get_exam_or_404(exam_id)

        if not exam["segmentation"]:
            abort(404)

        return send_file(
            svc.prediction_path(exam),
            as_attachment=True,
            download_name=f"exam{exam_id}_prediction.nii.gz",
        )

    @app.get("/analysis/<int:exam_id>/download/metadata")
    def download_metadata(exam_id):
        get_exam_or_404(exam_id)

        resp = app.response_class(
            json.dumps(
                svc.metadata_package(exam_id),
                indent=2,
                default=str,
            ),
            mimetype="application/json",
        )
        resp.headers["Content-Disposition"] = (
            f"attachment; filename=exam{exam_id}_metadata.json"
        )
        return resp

    @app.get("/history")
    def history():
        return render_template(
            "history.html",
            exams=db.get_all_examinations(),
        )

    @app.get("/history/<int:exam_id>")
    def history_detail(exam_id):
        get_exam_or_404(exam_id)
        return redirect(url_for("results", exam_id=exam_id))

    @app.post("/history/<int:exam_id>/delete")
    def history_delete(exam_id):
        get_exam_or_404(exam_id)
        res = svc.delete_examination(exam_id)

        flash(
            res.get("warning")
            or (
                "Examination deleted."
                if res["ok"]
                else res["error"]
            ),
            "warning"
            if res.get("warning")
            else ("success" if res["ok"] else "danger"),
        )
        return redirect(url_for("history"))

    def _latest_exam():
        rows = db.get_all_examinations()
        return db.get_examination(rows[0]["id"]) if rows else None

    @app.get("/capabilities")
    def capabilities():
        return render_template("model_capabilities.html")

    @app.get("/classification")
    def classification():
        return render_template("classification.html")

    @app.get("/analysis/<int:exam_id>/explainability")
    def exam_explainability(exam_id):
        exam = get_exam_or_404(exam_id)
        return render_template(
            "exam_explainability.html",
            exam=exam,
        )

    @app.get("/explainability")
    def explainability():
        return render_template("explainability.html")

    @app.get("/performance")
    def performance():
        vf = config_class.VERIFICATION_DIR / "checkpoint_inspection.json"
        ver = (
            json.loads(vf.read_text(encoding="utf-8"))
            if vf.is_file()
            else None
        )

        return render_template(
            "performance.html",
            perf=db.performance_summary(),
            ver=ver,
            mc=config_class.model_config(),
        )

    @app.get("/about")
    def about():
        return render_template("about.html")

    # ---------------- Patient 055 3D Stack Demo ----------------

    @app.get("/demo/3d")
    def demo_3d_patient_055():
        return render_template("patient_055_3d.html")

    @app.get("/demo/3d/data")
    def demo_3d_patient_055_data():
        source = Path.home() / "Downloads" / "patient_055_3d_demo.npz"

        if not source.is_file():
            return jsonify(
                error="Patient 055 demo stack was not found in Downloads."
            ), 404

        try:
            from scipy import ndimage
            from skimage.measure import marching_cubes

            with np.load(source, allow_pickle=False) as data:
                images = data["images"]
                masks = data["masks"]
                indices = data["slice_indices"]

            if images.shape != (149, 4, 240, 240):
                return jsonify(
                    error="Unexpected image stack dimensions."
                ), 400

            if masks.shape != (149, 240, 240):
                return jsonify(
                    error="Unexpected mask stack dimensions."
                ), 400

            if len(indices) != 149 or not np.all(np.diff(indices) == 1):
                return jsonify(
                    error="Slice indices are not consecutive."
                ), 400

            surfaces = []

            for label_id in (1, 2, 3):
                volume = (masks == label_id).astype(np.float32)

                if int(volume.sum()) < 4:
                    continue

                smooth = ndimage.gaussian_filter(volume, sigma=0.6)

                vertices, faces, _, _ = marching_cubes(
                    smooth,
                    level=0.5,
                    spacing=(1, 1, 1),
                    step_size=2,
                )

                surfaces.append(
                    {
                        "label": label_id,
                        "voxels": int(volume.sum()),
                        "x": vertices[:, 2].round(2).tolist(),
                        "y": vertices[:, 1].round(2).tolist(),
                        "z": vertices[:, 0].round(2).tolist(),
                        "i": faces[:, 0].tolist(),
                        "j": faces[:, 1].tolist(),
                        "k": faces[:, 2].tolist(),
                    }
                )

            return jsonify(
                patient="055",
                dimensions=list(masks.shape),
                coordinate_units="voxel units; physical spacing not verified",
                surfaces=surfaces,
            )

        except Exception:
            log.exception("Patient 055 3D demo generation failed")
            return jsonify(
                error="Could not generate the 3D demonstration surfaces."
            ), 500

    # ---------------- Patient FLAIR 3D Viewer ----------------

    @app.get("/demo/patient-flair-3d")
    def patient_flair_3d():
        return render_template("patient_flair_3d.html")

    @app.get("/demo/patient-flair-3d/data")
    def patient_flair_3d_data():
        source = (
            Path(__file__).resolve().parent
            / "data"
            / "brats_preprocessed_100"
            / "images"
            / "BRATS_003_FLAIR.nii"
        )

        if not source.is_file():
            return jsonify(
                error=f"Patient 003 FLAIR file not found: {source}"
            ), 404

        try:
            import nibabel as nib
            from skimage.measure import marching_cubes

            nii = nib.load(str(source))
            volume = np.asarray(nii.dataobj, dtype=np.float32)

            if volume.ndim != 3:
                raise ValueError(
                    f"Expected a 3D FLAIR volume, got shape {volume.shape}"
                )

                        # Downsample for a lighter browser visualization.
            small = volume[::2, ::2, ::2]

            finite = small[np.isfinite(small)]
            if finite.size == 0:
                raise ValueError("FLAIR volume contains no finite values")

            # Exclude zero-valued background before creating the surface.
            foreground_values = np.abs(finite[np.abs(finite) > 1e-6])
            if foreground_values.size == 0:
                raise ValueError("The FLAIR volume appears to contain only background")

            threshold = float(np.percentile(foreground_values, 5))
            surface_mask = np.isfinite(small) & (np.abs(small) >= threshold)

            # Keep the largest connected foreground region and smooth small gaps.
            from scipy import ndimage

            labels, component_count = ndimage.label(surface_mask)
            if component_count == 0:
                raise ValueError("No foreground region found in the FLAIR volume")

            component_sizes = np.bincount(labels.ravel())
            component_sizes[0] = 0
            largest_component = int(component_sizes.argmax())
            surface_mask = labels == largest_component

            surface_mask = ndimage.binary_closing(
                surface_mask,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=2,
            )
            surface_mask = ndimage.binary_fill_holes(surface_mask)

            if not surface_mask.any():
                raise ValueError("Could not create a foreground surface")

            vertices, faces, _, _ = marching_cubes(
                surface_mask.astype(np.float32),
                level=0.5,
                step_size=2,
            )
            

            return jsonify(
                patient="003",
                modality="FLAIR",
                dimensions=list(volume.shape),
                display_dimensions=list(small.shape),
                threshold=threshold,
                x=vertices[:, 0].round(2).tolist(),
                y=vertices[:, 1].round(2).tolist(),
                z=vertices[:, 2].round(2).tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
            )

        except Exception:
            log.exception("Patient FLAIR 3D viewer failed")
            return jsonify(
                error="Could not generate the FLAIR 3D surface. Check the Flask terminal."
            ), 500

    # ---------------- 2D MRI Demo ----------------

    @app.get("/demo/2d")
    def demo_2d():
        dataset_root = (
            Path(__file__).resolve().parent
            / "data"
            / "brats2d_preprocessed_100"
        )
        patient_slices = _list_2d_demo_samples(dataset_root)

        return render_template(
            "2d_demo.html",
            patient_ids=list(patient_slices),
            patient_slice_indices={
                patient_id: list(slices)
                for patient_id, slices in patient_slices.items()
            },
        )

    @app.get("/demo/2d/image/<kind>")
    def demo_2d_image(kind):
        image_files = {
            "t1": "t1_demo.png",
            "t1ce": "t1ce_demo.png",
            "t2": "t2_demo.png",
            "flair": "flair_demo.png",
            "comparison": "2d_inference_demo.png",
            "overlay": "2d_prediction_overlay.png",
        }

        if kind not in image_files:
            abort(404)

        image_path = os.path.join(
            config_class.VERIFICATION_DIR,
            image_files[kind],
        )

        if not os.path.isfile(image_path):
            abort(404)

        return send_file(image_path, mimetype="image/png")

    @app.post("/demo/2d/run")
    def demo_2d_run():
        dataset_root = (
            Path(__file__).resolve().parent
            / "data"
            / "brats2d_preprocessed_100"
        )
        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "masks"
        patient_slices = _list_2d_demo_samples(dataset_root)

        try:
            patient_id = request.form.get("patient_id", "")
            slice_index = int(request.form.get("slice_index", ""))

            if (
                patient_id not in patient_slices
                or slice_index not in patient_slices[patient_id]
            ):
                raise ValueError(
                    "Select a valid patient and an available slice number."
                )

            sample_name = patient_slices[patient_id][slice_index]
            image_path = images_dir / sample_name
            mask_path = masks_dir / sample_name

            print("2D DEBUG patient_id:", repr(patient_id))
            print("2D DEBUG slice_index:", repr(slice_index))
            print("2D DEBUG images_dir:", images_dir.resolve())
            print("2D DEBUG masks_dir:", masks_dir.resolve())
            print("2D DEBUG sample_name:", repr(sample_name))
            print("2D DEBUG image exists:", image_path.is_file(), image_path)
            print("2D DEBUG mask exists:", mask_path.is_file(), mask_path)
            print("2D DEBUG patient_id:", repr(patient_id))
            print("2D DEBUG slice_index:", repr(slice_index))
            print("2D DEBUG images_dir:", images_dir.resolve())
            print("2D DEBUG masks_dir:", masks_dir.resolve())
            print("2D DEBUG sample_name:", repr(sample_name))
            print("2D DEBUG image exists:", image_path.is_file(), image_path)
            print("2D DEBUG mask exists:", mask_path.is_file(), mask_path)
            if not image_path.is_file() or not mask_path.is_file():
                raise ValueError(
                    "That slice or its matching mask was not found. "
                    "Check the patient ID and slice number."
                )

            image = np.load(image_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)

            if image.shape != (4, 240, 240):
                raise ValueError(
                    f"Expected image shape (4, 240, 240), received {image.shape}."
                )

            if mask.shape != (240, 240):
                raise ValueError(
                    f"Expected mask shape (240, 240), received {mask.shape}."
                )

            if not np.issubdtype(image.dtype, np.number) or not np.issubdtype(
                mask.dtype, np.number
            ):
                raise ValueError(
                    "MRI and mask must contain numeric values."
                )

            if not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError(
                    "MRI or mask contains invalid values."
                )

            if not np.isin(np.unique(mask), [0, 1, 2, 3]).all():
                raise ValueError(
                    "Mask labels must be 0, 1, 2, or 3."
                )

            # MRI channels are already preprocessed.
            image = image.astype(np.float32)
            mask = mask.astype(np.uint8)

            padded = np.pad(
                image,
                ((0, 0), (8, 8), (8, 8)),
                mode="constant",
            )

            model, device = load_model(
                config_class.CHECKPOINT_PATH,
                config_class.model_config(),
                device_preference=config_class.DEVICE,
            )

            prediction = predict_slice(model, padded, device)
            prediction = prediction[8:248, 8:248]

            # Experimental Grad-CAM heatmap.
            gradcam = make_segmentation_gradcam(model, padded, device)
            gradcam = gradcam[8:248, 8:248]
        

            # Calculate 2D tumour measurements.
            tumour_mask = prediction > 0
            labelled_mask = label(tumour_mask)
            regions = regionprops(labelled_mask)

            tumour_area = int(tumour_mask.sum())
            tumour_area_percentage = round(
                (tumour_area / tumour_mask.size) * 100,
                2,
            )
            region_count = len(regions)

            largest_region_area = (
                int(max(regions, key=lambda r: r.area).area)
                if regions
                else 0
            )

            total_perimeter = round(
                float(sum(r.perimeter for r in regions)),
                2,
            )

            largest_eccentricity = (
                round(
                    float(max(regions, key=lambda r: r.area).eccentricity),
                    4,
                )
                if regions
                else 0.0
            )

            # Calculate binary tumour segmentation metrics.
            gt_region = mask > 0
            pred_region = prediction > 0

            intersection = np.logical_and(
                gt_region,
                pred_region,
            ).sum()

            union = np.logical_or(
                gt_region,
                pred_region,
            ).sum()

            dice_score = (
                (2.0 * intersection)
                / (gt_region.sum() + pred_region.sum())
                if (gt_region.sum() + pred_region.sum()) > 0
                else None
            )

            iou_score = (
                intersection / union
                if union > 0
                else None
            )

            dice_score = (
                round(float(dice_score), 4)
                if dice_score is not None
                else None
            )
            iou_score = (
                round(float(iou_score), 4)
                if iou_score is not None
                else None
            )

            # Professional comparison: MRI with transparent tumour overlays.
            gt_region = mask > 0
            pred_region = prediction > 0

            fig, axes = plt.subplots(1, 3, figsize=(18, 7))

            # Original MRI.
            axes[0].imshow(image[0], cmap="gray")
            axes[0].set_title(
                "Original MRI",
                fontsize=15,
                fontweight="bold",
            )

            # Ground truth overlay.
            axes[1].imshow(image[0], cmap="gray")
            axes[1].contourf(
                gt_region.astype(float),
                levels=[0.5, 1],
                colors=["green"],
                alpha=0.40,
            )
            axes[1].contour(
                gt_region.astype(float),
                levels=[0.5],
                colors=["green"],
                linewidths=1.5,
            )
            axes[1].set_title(
                "Ground Truth",
                fontsize=15,
                fontweight="bold",
            )

            # AI prediction overlay.
            axes[2].imshow(image[0], cmap="gray")
            axes[2].contourf(
                pred_region.astype(float),
                levels=[0.5, 1],
                colors=["red"],
                alpha=0.40,
            )
            axes[2].contour(
                pred_region.astype(float),
                levels=[0.5],
                colors=["red"],
                linewidths=1.5,
            )
            axes[2].set_title(
                "AI Prediction (U-Net)",
                fontsize=15,
                fontweight="bold",
            )

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()

            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="png",
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            buffer.seek(0)

            comparison_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            # Professional MRI overlay visualization.
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            # Ground-truth tumour overlay.
            axes[0].imshow(image[0], cmap="gray")
            axes[0].imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="Greens",
                alpha=0.45,
                vmin=1,
                vmax=3,
            )
            axes[0].contour(
                (mask > 0).astype(float),
                levels=[0.5],
                colors=["lime"],
                linewidths=1.5,
            )
            axes[0].set_title(
                "Ground Truth Overlay",
                fontsize=15,
                fontweight="bold",
            )

            # AI-predicted tumour overlay.
            axes[1].imshow(image[0], cmap="gray")
            axes[1].imshow(
                np.ma.masked_where(prediction == 0, prediction),
                cmap="Reds",
                alpha=0.45,
                vmin=1,
                vmax=3,
            )
            axes[1].contour(
                (prediction > 0).astype(float),
                levels=[0.5],
                colors=["red"],
                linewidths=1.5,
            )
            axes[1].set_title(
                "AI Prediction Overlay",
                fontsize=15,
                fontweight="bold",
            )

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()

            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="png",
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            buffer.seek(0)

            overlay_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")
            
            # Create the Grad-CAM visualization.
            fig, ax = plt.subplots(figsize=(7, 7))

            ax.imshow(image[0], cmap="gray")
            ax.imshow(
            gradcam,
            cmap="magma",
            alpha=0.5,
            vmin=0,
            vmax=1,
            )

            ax.set_title("Experimental Grad-CAM Heatmap")
            ax.axis("off")

            fig.tight_layout()

            buffer = io.BytesIO()
            fig.savefig(
            buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            )
            plt.close(fig)
            buffer.seek(0)

            gradcam_image = base64.b64encode(
            buffer.getvalue()
            ).decode("utf-8")

            return render_template(
                "2d_demo.html",
                patient_ids=list(patient_slices),
                patient_slice_indices={
                    patient_id: list(slices)
                    for patient_id, slices in patient_slices.items()
                },
                comparison_image=comparison_image,
                overlay_image=overlay_image,
                prediction_classes=np.unique(prediction).tolist(),
                dice_score=dice_score,
                iou_score=iou_score,
                tumour_area=tumour_area,
                tumour_area_percentage=tumour_area_percentage,
                region_count=region_count,
                largest_region_area=largest_region_area,
                total_perimeter=total_perimeter,
                largest_eccentricity=largest_eccentricity,
            )

        except Exception as exc:
            log.exception("2D demo prediction failed")
            flash(
                f"Prediction failed: {type(exc).__name__}: {exc}",
                "danger",
            )
            return redirect(url_for("demo_2d"))

    # ---------------- Health Check ----------------

    @app.get("/health")
    def health():
        ck = config_class.CHECKPOINT_PATH.is_file()

        try:
            with db.connect() as c:
                c.execute("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False

        return jsonify(
            status="ok" if db_ok else "degraded",
            database=db_ok,
            checkpoint_present=ck,
            version=APP_VERSION,
        ), (200 if db_ok else 503)

    # ---------------- errors ----------------

    @app.errorhandler(404)
    def e404(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def e413(e):
        msg = f"Upload exceeds the {config_class.MAX_UPLOAD_MB} MB limit."

        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify(ok=False, errors=[msg]), 413

        return render_template(
            "errors/error.html",
            code=413,
            message=msg,
        ), 413

    @app.errorhandler(400)
    def e400(e):
        return (
            render_template(
                "errors/error.html",
                code=400,
                message=getattr(e, "description", "Bad request"),
            ),
            400,
        )

    @app.errorhandler(500)
    def e500(e):
        log.exception("Unhandled error")
        return (
            render_template(
                "errors/error.html",
                code=500,
                message="Internal server error. See server log.",
            ),
            500,
        )

    return app


if __name__ == "__main__":
    import os

    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True,
    )




