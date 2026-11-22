"""INTEGRATION tests: need the real checkpoint (set CHECKPOINT_PATH or place it at model/weights/).
They verify weight compatibility + that the model runs; input is SYNTHETIC noise, so they do NOT verify
real-MRI inference, preprocessing correctness or accuracy.  Run: pytest -m integration"""
import numpy as np
import pytest
from config import Config
from model.checkpoint_validation import validate_checkpoint
from model.inference import predict_volume
from model.model_loader import load_model

pytestmark = pytest.mark.integration
CK = Config.CHECKPOINT_PATH
needs_ck = pytest.mark.skipif(not CK.is_file(), reason="real checkpoint not found at CHECKPOINT_PATH")


@needs_ck
def test_real_checkpoint_strict_load():
    rep = validate_checkpoint(CK, Config.model_config())
    assert rep["checks"]["strict_load_ok"], rep["compatibility"]
    assert rep["checkpoint"]["inferred_in_channels"] == 4 and rep["checkpoint"]["inferred_num_classes"] == 4


@needs_ck
def test_real_checkpoint_forward_on_synthetic_noise():
    cfg = Config.model_config(); model, dev = load_model(CK, cfg, "cpu")
    x = np.random.default_rng(1).normal(size=(4, 64, 64, 2)).astype(np.float32)
    out, st = predict_volume(model, x, dev, 2)
    assert out.shape == (64, 64, 2) and set(st["classes_present"]) <= {0, 1, 2, 3}
