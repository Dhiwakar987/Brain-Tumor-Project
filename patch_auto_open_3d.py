from pathlib import Path

path = Path("templates/results.html")
original = path.read_text(encoding="utf-8")

old = "window.HAS_GT={{ 'true' if exam.has_ground_truth else 'false' }};</script>"
new = old.replace(
    "</script>",
    "window.AUTO_OPEN_3D={{ 'true' if auto_open_3d else 'false' }};</script>"
)

if original.count(old) != 1:
    raise SystemExit("Expected script declaration not found exactly once. No changes made.")

path.with_suffix(".html.bak").write_text(original, encoding="utf-8")
path.write_text(original.replace(old, new, 1), encoding="utf-8")

print("AUTO_OPEN_3D connected to Flask template.")
print("Backup updated: templates/results.html.bak")
