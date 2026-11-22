"""Shared fixtures. ALL volumes here are SYNTHETIC (not BraTS) and the 'checkpoint' fixture is a
RANDOMLY INITIALISED model - used only to test software paths, never to judge accuracy."""
import io
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402

SHAPE = (48, 48, 6)


def make_volume(shape=SHAPE, seed=0, zooms=(1.0, 1.0, 1.0), affine=None):
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=np.float32)
    x, y, z = np.indices(shape)
    brain = ((x - shape[0] / 2) ** 2 + (y - shape[1] / 2) ** 2) < (shape[0] * 0.4) ** 2
    data[brain] = rng.normal(100, 10, brain.sum()).clip(1)
    aff = np.diag([*zooms, 1.0]) if affine is None else affine
    return nib.Nifti1Image(data, aff)


def make_labels(shape=SHAPE, zooms=(1.0, 1.0, 1.0), empty=False, shift=0):
    lab = np.zeros(shape, dtype=np.uint8)
    if not empty:
        lab[12 + shift:32 + shift, 12:32, 1:5] = 2
        lab[16 + shift:28 + shift, 16:28, 2:5] = 1
        lab[20 + shift:25 + shift, 20:25, 2:4] = 4
    return nib.Nifti1Image(lab, np.diag([*zooms, 1.0]))


def to_bytes(img, gz=True):
    """Serialise a NIfTI image to bytes using a portable temp dir (works on Windows, no hard-coded /tmp)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / ("img.nii.gz" if gz else "img.nii")
        nib.save(img, str(f)); return f.read_bytes()


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory):
    import segmentation_models_pytorch as smp
    import torch
    torch.manual_seed(0)
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=4)
    p = tmp_path_factory.mktemp("ckpt") / "synthetic_random_weights.pth"
    torch.save(m.state_dict(), p)
    return p


@pytest.fixture()
def app(tmp_path, synthetic_checkpoint):
    from app import create_app

    class TestConfig(Config):
        TESTING = True
        WTF_CSRF_ENABLED = False
        SECRET_KEY = "test"
        UPLOAD_DIR = tmp_path / "uploads"
        OUTPUT_DIR = tmp_path / "outputs"
        DATABASE_PATH = tmp_path / "test.db"
        CHECKPOINT_PATH = synthetic_checkpoint
        DEVICE = "cpu"
        INFERENCE_BATCH_SIZE = 4
    return create_app(TestConfig)


@pytest.fixture()
def client(app):
    return app.test_client()


def upload_files(shape=SHAPE, with_gt=False, skip=(), mismatch=None, corrupt=None):
    d = {}
    for i, k in enumerate(("t1", "t1ce", "t2", "flair")):
        if k in skip: continue
        sh = shape
        img = make_volume(sh, seed=i)
        if mismatch == k: img = make_volume((shape[0], shape[1], shape[2] + 2), seed=i)
        d[k] = (io.BytesIO(to_bytes(img)), f"{k}.nii.gz")
    if corrupt:
        d[corrupt] = (io.BytesIO(b"this is not a nifti file at all" * 50), f"{corrupt}.nii.gz")
    if with_gt:
        d["ground_truth"] = (io.BytesIO(to_bytes(make_labels(shape, shift=2))), "gt.nii.gz")
    return d


def wait_job(app, client, exam_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/analysis/{exam_id}/status").get_json()
        if s["job"] and s["job"]["status"] in ("completed", "failed", "interrupted"):
            return s
        time.sleep(0.3)
    raise TimeoutError("job did not finish")
