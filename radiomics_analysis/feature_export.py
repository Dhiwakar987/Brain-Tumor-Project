"""CSV/Excel export and predicted-vs-reference feature comparison."""
from pathlib import Path

import numpy as np
import pandas as pd


def save_radiomics_csv(rows, path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_radiomics_excel(rows, path, settings=None) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="features", index=False)
        if settings is not None:
            pd.DataFrame([{"setting": k, "value": str(v)} for k, v in (settings.get("setting") or {}).items()]
                         ).to_excel(xw, sheet_name="settings", index=False)
    return path


def compare_radiomics_features(pred_rows, ref_rows) -> pd.DataFrame:
    """Paired comparison: SAME patient/scan/settings, features from predicted vs reference mask.
    Relative difference is None when the reference value is 0 (undefined)."""
    keys = ["region", "modality", "feature_class", "feature_name"]
    p = pd.DataFrame(pred_rows)[keys + ["value"]].rename(columns={"value": "predicted_value"})
    r = pd.DataFrame(ref_rows)[keys + ["value"]].rename(columns={"value": "reference_value"})
    m = r.merge(p, on=keys, how="outer")
    m["abs_diff"] = (m["predicted_value"] - m["reference_value"]).abs()
    m["rel_diff_percent"] = np.where(m["reference_value"].abs() > 0,
                                     100.0 * (m["predicted_value"] - m["reference_value"]) / m["reference_value"].abs(), np.nan)
    return m
