"""Central configuration. Every value can be overridden with an environment variable
(see .env.example). No local machine paths are hard-coded: relative paths are resolved
against the project folder."""
import os
import secrets
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
APP_VERSION = "2.0.0"


def _load_dotenv():
    """Tiny .env reader (no extra dependency). Existing environment variables win."""
    f = BASE_DIR / ".env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: Notepad adds a BOM that would corrupt the first key
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default)).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Config:
    # --- Flask / security -------------------------------------------------
    # If SECRET_KEY is not set a random one is generated per process (sessions/CSRF
    # tokens then reset on restart). Set SECRET_KEY in .env for stable sessions.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    SECRET_KEY_WAS_GENERATED = "SECRET_KEY" not in os.environ
    WTF_CSRF_TIME_LIMIT = None
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "800"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = tuple(
        e.strip().lower() for e in os.environ.get("ALLOWED_EXTENSIONS", ".nii,.nii.gz").split(",")
    )
    # --- Paths ---------------------------------------------------------------
    UPLOAD_DIR = _path("UPLOAD_DIR", "uploads")
    OUTPUT_DIR = _path("OUTPUT_DIR", "outputs")
    DATABASE_PATH = _path("DATABASE_PATH", "outputs/app.db")
    VERIFICATION_DIR = _path("VERIFICATION_DIR", "verification")
    BACKUP_DIR = _path("BACKUP_DIR", "backups")
    CHECKPOINT_PATH = _path("CHECKPOINT_PATH", "model/weights/best_unet_brats.pth")
    MODEL_CONFIG_PATH = _path("MODEL_CONFIG_PATH", "model/model_config.yaml")
    RADIOMICS_SETTINGS_PATH = _path("RADIOMICS_SETTINGS_PATH", "radiomics_analysis/settings.yaml")
    BRATS_DATASET_PATH = os.environ.get("BRATS_DATASET_PATH", "")  # used by experiment scripts only
    # --- Runtime ---------------------------------------------------------------
    DEVICE = os.environ.get("DEVICE", "auto")  # auto | cpu | cuda
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    DATA_RETENTION_DAYS = int(os.environ.get("DATA_RETENTION_DAYS", "0"))  # 0 = keep forever
    ALLOW_UNSAFE_CHECKPOINT_PICKLE = _bool("ALLOW_UNSAFE_CHECKPOINT_PICKLE", False)
    INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "8"))
    TESTING = False

    @classmethod
    def model_config(cls) -> dict:
        return load_yaml(cls.MODEL_CONFIG_PATH)

    @classmethod
    def radiomics_settings(cls) -> dict:
        return load_yaml(cls.RADIOMICS_SETTINGS_PATH)
