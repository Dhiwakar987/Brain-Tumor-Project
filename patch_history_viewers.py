from pathlib import Path

path = Path("templates/history.html")
original = path.read_text(encoding="utf-8")

old = '''<td class="text-nowrap"><a class="btn btn-sm btn-outline-primary" href="{{ url_for('history_detail', exam_id=e.id) }}">Open</a>
{% if e.n_reports %}<a class="btn btn-sm btn-outline-secondary" href="{{ url_for('download_report', exam_id=e.id) }}">PDF</a>{% endif %}'''

new = '''<td class="text-nowrap"><a class="btn btn-sm btn-outline-primary" href="{{ url_for('history_detail', exam_id=e.id) }}">Open</a>
<a class="btn btn-sm btn-outline-info" href="{{ url_for('viewer_2d', exam_id=e.id) }}">2D Viewer</a>
<a class="btn btn-sm btn-outline-info" href="{{ url_for('viewer_3d', exam_id=e.id) }}">3D Viewer</a>
{% if e.n_reports %}<a class="btn btn-sm btn-outline-secondary" href="{{ url_for('download_report', exam_id=e.id) }}">PDF</a>{% endif %}'''

if original.count(old) != 1:
    raise SystemExit("Expected History action section was not found exactly once. No changes made.")

path.with_suffix(".html.bak").write_text(original, encoding="utf-8")
path.write_text(original.replace(old, new, 1), encoding="utf-8")

print("History updated: Open, 2D Viewer, 3D Viewer, PDF, Delete.")
print("Backup created: templates/history.html.bak")
