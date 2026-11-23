from pathlib import Path

path = Path("static/js/viewer.js")
original = path.read_text(encoding="utf-8")

old = """    $('vModality').addEventListener('change', loadMri3d);
    loadMri3d();
    return;
"""

new = """    $('vModality').addEventListener('change', loadMri3d);
    loadMri3d();

    // Open the 3D MRI planes when navigation requested the 3D view.
    if (window.AUTO_OPEN_3D) {
      const tabButton = document.querySelector('[data-bs-target="#tab3d"]');
      if (tabButton && window.bootstrap) {
        window.bootstrap.Tab.getOrCreateInstance(tabButton).show();
      }
    }
    return;
"""

if original.count(old) != 1:
    raise SystemExit("Expected pre-segmentation viewer block not found exactly once. No changes made.")

path.with_suffix(".js.bak").write_text(original, encoding="utf-8")
path.write_text(original.replace(old, new, 1), encoding="utf-8")

print("Pre-segmentation 3D tab activation added.")
print("Backup created: static/js/viewer.js.bak")
