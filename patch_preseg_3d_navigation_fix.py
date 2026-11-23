from pathlib import Path

p = Path("static/js/viewer.js")
s = p.read_text(encoding="utf-8")

old = """    // Open the 3D MRI planes when navigation requested the 3D view.
    if (window.AUTO_OPEN_3D) {
      const tabButton = document.querySelector('[data-bs-target="#tab3d"]');
      if (tabButton && window.bootstrap) {
        window.bootstrap.Tab.getOrCreateInstance(tabButton).show();
      }
    }
"""

new = """    // Navigate to the pre-segmentation 3D MRI planes panel.
    if (window.AUTO_OPEN_3D) {
      requestAnimationFrame(() => {
        p3.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
"""

if s.count(old) != 1:
    raise SystemExit("Expected navigation block not found exactly once. No changes made.")

p.with_suffix(".js.navigation.bak").write_text(s, encoding="utf-8")
p.write_text(s.replace(old, new, 1), encoding="utf-8")

print("Pre-segmentation 3D navigation corrected.")
print("Backup created: static/js/viewer.js.navigation.bak")
