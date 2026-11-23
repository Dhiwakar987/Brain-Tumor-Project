from pathlib import Path
import re

app_path = Path("app.py")
base_path = Path("templates/base.html")

app = app_path.read_text(encoding="utf-8")
base = base_path.read_text(encoding="utf-8")

# Replace the existing 2D/3D route block.
pattern = re.compile(
    r'(?ms)^    @app\.get\("/viewer/2d"\).*?'
    r'(?=^    @app\.get\("/analysis/<int:exam_id>/slice"\))'
)

replacement = '''    @app.get("/viewer/2d")
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

'''

updated_app, count = pattern.subn(replacement, app, count=1)
if count != 1:
    raise SystemExit("Could not find the expected viewer route block in app.py; no files changed.")

# Make sidebar viewer links use the current exam ID when the template has one.
old_link = 'href="{{ url_for(ep) }}"'
new_link = '''href="{{ url_for(ep, exam_id=exam.id) if exam is defined and exam and exam.id is defined and ep in ['viewer_2d', 'viewer_3d'] else url_for(ep) }}"'''

if old_link not in base:
    raise SystemExit("Could not find the expected sidebar link in base.html; no files changed.")

# Back up both files before writing.
app_path.with_suffix(".py.bak").write_text(app, encoding="utf-8")
base_path.with_suffix(".html.bak").write_text(base, encoding="utf-8")

base = base.replace(old_link, new_link, 1)

app_path.write_text(updated_app, encoding="utf-8")
base_path.write_text(base, encoding="utf-8")

print("Updated viewer routes and exam-aware sidebar links.")
print("Backups created: app.py.bak and templates/base.html.bak")
