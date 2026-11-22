# Installation
Python 3.10–3.12. Windows/Linux.
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
# GPU (example CUDA 12.1): pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt
```
### PyRadiomics troubleshooting
`pip install pyradiomics` can fail ("inconsistent version" metadata / build error) on Python 3.12. Options: (a) use Python 3.10/3.11 where wheels are usually available; (b) `conda install -c conda-forge pyradiomics`; (c) source build: `pip install setuptools_scm scikit-build-core cmake ninja pybind11 cython wheel` then `pip install --no-build-isolation https://github.com/AIM-Harvard/pyradiomics/archive/refs/heads/master.zip` (needs a C/C++ compiler). The development environment of this project used option (c) (PyRadiomics 3.1.1.dev111).
### Checkpoint
Copy `best_unet_brats.pth` to `model/weights/best_unet_brats.pth` (never overwritten by the app) or set `CHECKPOINT_PATH`. Then `python scripts/verify_checkpoint.py`.
### Run
`python app.py` → http://127.0.0.1:5000 (`HOST`, `PORT` env vars). For production-like use run behind a real WSGI server with a *single process* (job queue is in-process) and set `SECRET_KEY`.
### Backup / restore
```bash
python scripts/backup_restore.py backup          # online SQLite snapshot + uploads + outputs -> backups/*.zip, then verifies
python scripts/backup_restore.py verify backups/backup_X.zip
python scripts/backup_restore.py restore backups/backup_X.zip --target restored_data    # refuses non-empty targets
python scripts/backup_restore.py check           # detect missing files / DB integrity
```
A backup is only considered good if `verify` reports `ok: true`.
