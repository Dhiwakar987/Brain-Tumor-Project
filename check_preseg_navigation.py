from pathlib import Path

s = Path("static/js/viewer.js").read_text(encoding="utf-8")

assert "p3.scrollIntoView({ behavior: 'smooth', block: 'start' })" in s
assert "tabButton" not in s

print("Pre-segmentation navigation check passed.")
