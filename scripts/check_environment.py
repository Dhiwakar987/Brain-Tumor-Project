"""Diagnose the Python environment (NumPy / PyRadiomics / PyTorch compatibility) and print fixes.
Run:  python scripts/check_environment.py"""
import importlib, platform, sys

print(f"Python {platform.python_version()} on {platform.platform()}")
ok = True
for name in ("numpy", "scipy", "pandas", "nibabel", "SimpleITK", "torch", "segmentation_models_pytorch", "radiomics", "flask", "flask_wtf", "reportlab", "skimage", "openpyxl", "yaml"):
    try:
        m = importlib.import_module(name); print(f"  OK   {name:30s} {getattr(m, '__version__', '')}")
    except Exception as e:  # ImportError, or 'numpy.core.multiarray failed to import' from a binary built against another NumPy
        ok = False; print(f"  FAIL {name:30s} {type(e).__name__}: {str(e)[:110]}")
        if "numpy" in str(e).lower():
            print("       -> A compiled package was built for a different NumPy major version.\n"
                  '          Fix (PowerShell):  pip install --upgrade --force-reinstall "numpy<2"  then reinstall the failing package;\n'
                  "          or use Python 3.11 in a fresh virtual environment and `pip install -r requirements.txt`.")
if ok:
    try:
        import numpy as np, SimpleITK as sitk
        from radiomics import featureextractor
        img = sitk.GetImageFromArray(np.random.default_rng(0).random((12, 12, 12)).astype("float32")); m = np.zeros((12, 12, 12), "uint8"); m[3:9, 3:9, 3:9] = 1
        r = featureextractor.RadiomicsFeatureExtractor({"featureClass": {"firstorder": []}}).execute(img, sitk.GetImageFromArray(m))
        print("  OK   PyRadiomics end-to-end smoke test (first-order features):", sum(1 for k in r if k.startswith('original_')), "values")
    except Exception as e:
        ok = False; print("  FAIL PyRadiomics smoke test:", type(e).__name__, str(e)[:150])
sys.exit(0 if ok else 1)
