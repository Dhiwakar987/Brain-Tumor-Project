"""Matplotlib figures (used in the PDF report) - drawn from real saved data only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .mri_viewer import best_slice_index, display_scale, display_window, extract_slice, inplane_aspect

CLASS_COLORS = ["#00000000", "#e53935", "#fdd835", "#1e88e5"]  # 1 NCR red, 2 ED yellow, 3(=label 4) ET blue


def snapshot_figure(image3d, labels3d, zooms, path, title="", view="axial"):
    idx = best_slice_index(labels3d, view)
    base = display_scale(extract_slice(image3d, view, idx), display_window(image3d))
    lab = extract_slice(labels3d, view, idx)
    show = np.zeros_like(lab, dtype=np.uint8); show[lab == 1] = 1; show[lab == 2] = 2; show[lab == 4] = 3
    asp = inplane_aspect(zooms, view)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6), dpi=130)
    for a in ax: a.axis("off")
    ax[0].imshow(base, cmap="gray", origin="lower", aspect=asp); ax[0].set_title(f"{title} ({view}, slice {idx})", fontsize=7)
    ax[1].imshow(base, cmap="gray", origin="lower", aspect=asp)
    ax[1].imshow(np.ma.masked_equal(show, 0), cmap=ListedColormap(CLASS_COLORS[1:]), vmin=1, vmax=3, alpha=0.55, origin="lower", aspect=asp, interpolation="nearest")
    ax[1].set_title("Predicted overlay: red=NCR/NET, yellow=edema, blue=ET", fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def metrics_bar(metrics: dict, path):
    regs = list(metrics.keys())
    vals = [metrics[r]["dice"] if metrics[r]["dice"] is not None else np.nan for r in regs]
    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=130)
    ax.bar(regs, np.nan_to_num(vals), color="#1e88e5")
    for i, v in enumerate(vals):
        ax.text(i, 0.02, "N/A" if np.isnan(v) else f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Dice (this examination)"); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def three_panel_figure(image3d, pred3d, gt3d, zooms, path, view="axial"):
    """Original MRI | Ground truth | AI prediction on the SAME slice (slice with most predicted tumour; GT panel only if GT exists)."""
    idx = best_slice_index(pred3d if pred3d.any() else (gt3d if gt3d is not None else pred3d), view)
    base = display_scale(extract_slice(image3d, view, idx), display_window(image3d)); asp = inplane_aspect(zooms, view)
    def show(lab3):
        lab = extract_slice(lab3, view, idx); o = np.zeros_like(lab, dtype=np.uint8); o[lab == 1] = 1; o[lab == 2] = 2; o[lab == 4] = 3; return o
    panels = [("Original MRI", None)] + ([("Ground Truth", gt3d)] if gt3d is not None else []) + [("AI Prediction", pred3d)]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), dpi=130)
    ax = np.atleast_1d(ax)
    for a, (title, lab3) in zip(ax, panels):
        a.imshow(base, cmap="gray", origin="lower", aspect=asp); a.axis("off"); a.set_title(f"{title} ({view} {idx})", fontsize=8)
        if lab3 is not None:
            a.imshow(np.ma.masked_equal(show(lab3), 0), cmap=ListedColormap(CLASS_COLORS[1:]), vmin=1, vmax=3, alpha=0.55, origin="lower", aspect=asp, interpolation="nearest")
    if gt3d is None: fig.text(0.5, 0.01, "Ground truth not provided - comparison unavailable", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path
