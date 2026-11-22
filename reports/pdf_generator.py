"""PDF report built with ReportLab from SAVED examination data only."""
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DISCLAIMER = ("RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE. This report is produced by an academic software prototype. "
              "Segmentations and radiomics features have not been clinically validated and must not be used for diagnosis, "
              "treatment decisions, or prognosis.")


def _fmt(v, nd=4):
    return "N/A" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8),
          ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eaf3")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t


def generate_pdf_report(out_path, exam: dict, radiomics_rows: list, env_info: dict, snapshot_path=None,
                        metrics_chart_path=None, app_version="1.0.0", model_info=None, measurements=None, three_panel_path=None) -> Path:
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8, leading=10)
    warn = ParagraphStyle("warn", parent=small, textColor=colors.HexColor("#b71c1c"))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Brain Tumor Segmentation and Radiomics Report")
    S = []
    S += [Paragraph("Web-Based Brain Tumor Segmentation and Radiomics Analysis Using Deep Learning", ss["Title"]),
          Paragraph(DISCLAIMER, warn), Spacer(1, 8)]
    S += [Paragraph("1. Examination", ss["Heading2"]), _table([
        ["Examination ID", str(exam["id"])], ["Reference (opaque UID)", exam["exam_uid"][:12] + "..."],
        ["Patient label (de-identified)", exam.get("patient_label") or "-"], ["Created (UTC)", exam["created_at"]],
        ["Report generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")], ["Status", exam["status"]],
        ["Ground truth supplied", "Yes" if exam["has_ground_truth"] else "No"],
        ["Human review status", exam["review_status"].replace("_", " ") + " (not clinical approval)"]], [5 * cm, 12 * cm], header=False)]
    S += [Paragraph("2. Input modalities", ss["Heading2"]), _table(
        [["Role", "Original name", "Size (bytes)", "SHA-256 (first 16)"]] +
        [[f["role"], f["original_name"], f["size_bytes"], (f["sha256"] or "")[:16]] for f in exam["files"]])]
    v = (exam.get("validation") or {})
    if v:
        S += [Paragraph("Input validation: " + ("passed" if v.get("ok") else "FAILED") + f" | details: {v.get('details')}", small)]
        for w in v.get("warnings", []): S += [Paragraph("Warning: " + w, warn)]
    cfg = exam.get("config_snapshot") or {}
    S += [Paragraph("3. Model and preprocessing", ss["Heading2"]), _table([
        ["Architecture", f"U-Net / {cfg.get('encoder_name')} ({cfg.get('in_channels')} in, {cfg.get('classes')} classes)"],
        ["Checkpoint SHA-256", exam.get("checkpoint_sha256") or "-"], ["Channel order", str(cfg.get("modality_order"))],
        ["Preprocessing", str(cfg.get("preprocessing"))], ["Class mapping (model->BraTS)", str(cfg.get("class_mapping"))],
        ["Verification flags", str(cfg.get("verification"))]], [5 * cm, 12 * cm], header=False)]
    if not (cfg.get("verification") or {}).get("preprocessing_verified_against_training"):
        S += [Paragraph("Preprocessing and label mapping were NOT verified against the original training pipeline.", warn)]
    seg = exam.get("segmentation")
    S += [Paragraph("4. Segmentation summary", ss["Heading2"])]
    if seg:
        S += [Paragraph(f"Device: {seg['device']} | Inference: {seg['inference'] and seg['inference']['stats']['elapsed_s']} s", small),
              _table([["Region", "Voxels", "Volume (mL)"]] + [[r, seg["region_voxels"][r], _fmt(seg["region_volumes"][r], 3)] for r in ("WT", "TC", "ET")]),
              Paragraph("Volumes = voxels x voxel volume (mm3) / 1000. Not a clinical measurement.", small)]
        for w in (seg.get("qc") or []): S += [Paragraph("QC warning: " + w, warn)]
        if snapshot_path and Path(snapshot_path).exists():
            S += [Spacer(1, 6), Image(str(snapshot_path), width=16 * cm, height=8.2 * cm)]
    else:
        S += [Paragraph("No segmentation has been completed for this examination.", small)]
    if measurements:
        rows = [["Region", "Voxels", "Volume (mm3)", "Volume (mL)", "Bbox extent (mm)", "Surface (mm2)", "Sphericity", "Max 3D diam. (mm)"]]
        for r in ("WT", "TC", "ET"):
            d = measurements["predicted"][r]; ext = d["bbox_extent_mm"]
            rows.append([r, d["voxel_count"], _fmt(d["volume_mm3"], 1), _fmt(d["volume_ml"], 3), "-" if not ext else " x ".join(f"{v:.0f}" for v in ext),
                         _fmt(d["surface_area_mm2"], 1), _fmt(d["sphericity"], 3), _fmt(d["max_3d_diameter_mm"], 1)])
        S += [Spacer(1, 6), Paragraph("Tumour measurements (predicted mask, physical units from the NIfTI header)", small), _table(rows),
              Paragraph("Bbox extent is along the image array axes. Surface-based values use a marching-cubes mesh; PyRadiomics shape features (below) are authoritative for radiomics.", small)]
    if three_panel_path and Path(three_panel_path).exists():
        S += [Spacer(1, 6), Paragraph("Visual comparison (same slice)", small), Image(str(three_panel_path), width=17 * cm, height=6 * cm if exam["has_ground_truth"] else 5.4 * cm)]
    S += [Paragraph("5. Radiomics summary", ss["Heading2"])]
    if radiomics_rows:
        import pandas as pd
        df = pd.DataFrame(radiomics_rows)
        summ = df.groupby(["segmentation_source", "region", "modality"]).size().reset_index(name="n_features")
        S += [_table([["Source", "Region", "Modality", "# features"]] + summ.values.tolist())]
        keys = ["original_shape_VoxelVolume"]
        sh = df[(df.feature_class == "shape") & (df.feature_name.isin(["VoxelVolume", "Sphericity", "Maximum3DDiameter"]))]
        if len(sh):
            S += [Spacer(1, 4), _table([["Region", "Shape feature", "Value"]] + [[r.region, r.feature_name, _fmt(float(r.value), 4)] for r in sh.itertuples()])]
        runs = exam.get("radiomics_runs") or []
        if runs:
            S += [Paragraph(f"PyRadiomics {runs[0]['pyradiomics_version']}. Settings: {runs[0]['settings_json'][:600]}", small)]
        S += [Paragraph("Radiomics features are descriptive research quantities, not validated biomarkers. Full tables: CSV export. "
                        "No claim of full IBSI compliance is made.", small)]
    else:
        S += [Paragraph("Radiomics has not been run for this examination.", small)]
    S += [Paragraph("6. Evaluation", ss["Heading2"])]
    if exam.get("evaluation"):
        S += [_table([["Region", "Dice", "IoU", "Precision", "Recall", "Specificity", "HD95 (mm)"]] +
                     [[e["region"], _fmt(e["dice"]), _fmt(e["iou"]), _fmt(e["precision"]), _fmt(e["recall"]), _fmt(e["specificity"]), _fmt(e["hd95_mm"], 2)] for e in exam["evaluation"]]),
              Paragraph("Policy: " + (exam["evaluation"][0].get("policy") or ""), small)]
        if metrics_chart_path and Path(metrics_chart_path).exists():
            S += [Image(str(metrics_chart_path), width=8 * cm, height=5.2 * cm)]
    else:
        S += [Paragraph("Quantitative accuracy not available: no ground-truth mask was supplied / evaluation not run.", small)]
    S += [Paragraph("Classification and explainability", ss["Heading3"]),
          Paragraph("Not available: this project contains no trained tumour-classification model, no Grad-CAM++ and no TreeSHAP implementation. No classification or explanation is shown rather than a fabricated one.", small)]
    S += [Paragraph("7. Reproducibility and software", ss["Heading2"]), _table([[k, str(v)] for k, v in {
        "Application version": app_version, **env_info}.items()], [5 * cm, 12 * cm], header=False)]
    if exam.get("timings"):
        S += [Paragraph("Timings (s): " + str(exam["timings"]), small)]
    S += [Spacer(1, 10), Paragraph(DISCLAIMER, warn)]
    def _pn(canvas, d):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.grey)
        canvas.drawString(1.6 * cm, 0.9 * cm, "Research prototype - not a medical device")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Page {d.page}"); canvas.restoreState()
    doc.build(S, onFirstPage=_pn, onLaterPages=_pn)
    return out_path
