"""NIfTI reading/writing helpers. Images are never silently reoriented or resampled."""
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiError(ValueError):
    """Raised for unreadable / invalid NIfTI input."""


@dataclass
class NiftiVolume:
    data: np.ndarray          # 3-D float32 (or label array)
    affine: np.ndarray        # 4x4
    zooms: tuple              # voxel spacing in mm
    shape: tuple
    axcodes: tuple            # orientation codes, e.g. ('R','A','S')


def load_nifti(path, dtype=np.float32) -> NiftiVolume:
    """Load a 3-D NIfTI. 4-D files with a singleton last axis are squeezed."""
    try:
        img = nib.load(str(path))
        data = np.array(img.dataobj)  # real copy: no open memmap, so Windows can delete the file later
    except Exception as exc:  # corrupt gzip, wrong magic, etc.
        raise NiftiError(f"Not a readable NIfTI file: {exc}") from exc
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise NiftiError(f"Expected a 3-D volume, got {data.ndim}-D with shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return NiftiVolume(
        data=data.astype(dtype, copy=False), affine=np.array(img.affine, dtype=np.float64),
        zooms=zooms, shape=tuple(int(s) for s in data.shape),
        axcodes=tuple(nib.aff2axcodes(img.affine)),
    )


def save_nifti_like(array: np.ndarray, reference_path, out_path, dtype=np.uint8) -> Path:
    """Save `array` using the affine + header of the reference NIfTI (geometry preserved).
    Written to a temporary 'partial_*' file first so interrupted runs never leave a
    half-written output under the final name."""
    ref = nib.load(str(reference_path))
    header = ref.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.asarray(array).astype(dtype), ref.affine, header)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("partial_" + out_path.name)  # keeps .nii.gz so nibabel picks the right writer
    nib.save(out, str(tmp))
    tmp.replace(out_path)
    return out_path


def verify_saved_geometry(saved_path, reference_path, atol=1e-5) -> dict:
    """Read a saved NIfTI back and compare shape/affine with the reference."""
    a, b = nib.load(str(saved_path)), nib.load(str(reference_path))
    return {
        "shape_match": tuple(a.shape) == tuple(b.shape),
        "affine_match": bool(np.allclose(a.affine, b.affine, atol=atol)),
        "readable": True,
    }


def to_sitk(array: np.ndarray, affine: np.ndarray, zooms):
    """Convert a NIfTI array (i,j,k) + RAS affine to a SimpleITK image (LPS physical space)."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array, (2, 1, 0))))
    zooms = np.asarray(zooms, dtype=float)
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    direction = ras2lps @ (affine[:3, :3] / zooms)
    origin = ras2lps @ affine[:3, 3]
    img.SetSpacing(tuple(float(z) for z in zooms))
    img.SetOrigin(tuple(float(o) for o in origin))
    img.SetDirection(tuple(float(d) for d in direction.flatten()))
    return img
