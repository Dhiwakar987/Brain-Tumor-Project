"""Secure upload handling: extension whitelist, magic-byte check, unique server-side names,
path-traversal-proof storage, hashing and cleanup."""
import gzip
import hashlib
import shutil
from pathlib import Path

from werkzeug.utils import secure_filename

MODALITIES = ("t1", "t1ce", "t2", "flair")
ROLES = MODALITIES + ("ground_truth",)


class UploadError(ValueError):
    pass


def allowed_extension(filename: str, allowed) -> str | None:
    """Return the matched extension (.nii.gz / .nii) or None."""
    low = (filename or "").lower()
    for ext in sorted(allowed, key=len, reverse=True):
        if low.endswith(ext) and len(low) > len(ext):
            return ext
    return None


def sniff_nifti(path: Path, ext: str):
    """Cheap content check so that renamed executables/other files are rejected before nibabel
    ever touches them. .nii.gz must start with gzip magic; NIfTI-1 header has sizeof_hdr=348 and
    magic 'n+1'/'ni1'; NIfTI-2 sizeof_hdr=540."""
    with open(path, "rb") as fh:
        head = fh.read(2)
    if ext.endswith(".gz"):
        if head != b"\x1f\x8b":
            raise UploadError("File has a .nii.gz name but is not gzip-compressed")
        try:
            with gzip.open(path, "rb") as g:
                hdr = g.read(348)
        except Exception as exc:
            raise UploadError(f"Corrupt gzip stream: {exc}")
    else:
        with open(path, "rb") as fh:
            hdr = fh.read(348)
    if len(hdr) < 348:
        raise UploadError("File too small to be a NIfTI volume")
    sizeof_hdr = int.from_bytes(hdr[:4], "little"), int.from_bytes(hdr[:4], "big")
    if 348 not in sizeof_hdr and 540 not in sizeof_hdr:
        raise UploadError("File does not look like a NIfTI file (bad header)")


def sha256_of(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_upload(storage, role: str, dest_dir: Path, allowed, ) -> dict:
    """Save one werkzeug FileStorage to dest_dir under a server-chosen name '<role><ext>'.
    The user's filename is only kept as display metadata (sanitised)."""
    if role not in ROLES:
        raise UploadError(f"Unknown role {role}")
    original = secure_filename(storage.filename or "")
    ext = allowed_extension(original, allowed)
    if not ext:
        raise UploadError(f"{role.upper()}: '{original or 'file'}' must be a .nii or .nii.gz file")
    dest_dir.mkdir(parents=True, exist_ok=True)
    stored = f"{role}{ext}"
    target = (dest_dir / stored).resolve()
    if dest_dir.resolve() not in target.parents:
        raise UploadError("Invalid storage path")
    storage.save(str(target))
    size = target.stat().st_size
    if size == 0:
        target.unlink(missing_ok=True); raise UploadError(f"{role.upper()}: file is empty")
    try:
        sniff_nifti(target, ext)
    except UploadError:
        target.unlink(missing_ok=True); raise
    return {"role": role, "original_name": original, "stored_name": stored, "size_bytes": size, "sha256": sha256_of(target)}


def exam_dir(base: Path, uid: str) -> Path:
    """Resolve a per-examination folder; uid must be hex."""
    if not uid or not all(c in "0123456789abcdef" for c in uid):
        raise ValueError("Invalid examination uid")
    return base / uid


def remove_tree(path: Path) -> bool:
    """Delete a directory; returns False (and does not raise) on partial failure."""
    try:
        if path.exists():
            shutil.rmtree(path)
        return True
    except OSError:
        return False


def cleanup_partial_outputs(output_dir: Path) -> int:
    n = 0
    for p in output_dir.rglob("partial_*"):
        try: p.unlink(); n += 1
        except OSError: pass
    return n
