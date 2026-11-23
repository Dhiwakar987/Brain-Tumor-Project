from pathlib import Path

path = Path("templates/results.html")
text = path.read_text(encoding="utf-8")

replacements = [
    (
        '<div class="fw-semibold mb-1">2D MRI viewer</div>\n'
        '    <div class="small text-muted">MRI-only view. Run segmentation to enable overlays, comparison, and 3D tumour surfaces.</div>',
        '<div class="fw-semibold mb-1">2D MRI viewer — capabilities</div>\n'
        '    <div class="small text-muted">Explore this examination in axial, coronal, and sagittal planes. Select T1, T1CE, T2, or FLAIR; move through slices; zoom and pan. After segmentation, the viewer can also display the AI mask over the MRI or show the mask alone. Ground-truth comparison is available only when a reference mask was provided.</div>'
    ),
    (
        '<div class="card-header">3D MRI view</div>\n'
        '  <div class="card-body">\n'
        '    <div class="small text-muted mb-2">Three intersecting MRI planes. Change the modality in the 2D viewer above to update this view.</div>',
        '<div class="card-header">3D MRI viewer — capabilities</div>\n'
        '  <div class="card-body">\n'
        '    <div class="small text-muted mb-2">View this examination as three intersecting MRI planes. Change the modality in the 2D viewer above to update the planes. After segmentation, use the 3D brain tab to rotate the reconstructed brain and inspect predicted tumour regions. This MRI-plane display is not itself a tumour surface.</div>'
    ),
    (
        '<div class="tab-pane fade show active" id="tab2d">',
        '<div class="tab-pane fade show active" id="tab2d">\n'
        '   <div class="alert alert-info small py-2">2D capabilities: change MRI modality and plane, scroll through slices, zoom/pan, and view MRI with or without the predicted mask. Ground-truth comparison requires an uploaded reference mask.</div>'
    ),
    (
        '<div class="tab-pane fade" id="tab3d">',
        '<div class="tab-pane fade" id="tab3d">\n'
        '   <div class="alert alert-info small py-2">3D capabilities: rotate the reconstruction, zoom, reset the camera, and toggle the brain surface or edema. This reconstruction uses the selected examination’s MRI and segmentation; it is available only after segmentation succeeds.</div>'
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"Expected one matching section, found {count}. "
            "No file was changed; check the template text."
        )
    text = text.replace(old, new, 1)

path.with_suffix(".html.bak").write_text(
    path.read_text(encoding="utf-8"),
    encoding="utf-8"
)
path.write_text(text, encoding="utf-8")
print("Added 2D and 3D capability descriptions. Backup: templates/results.html.bak")
