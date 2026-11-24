from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_segmentation_gradcam(
    model: torch.nn.Module,
    slice_chw: np.ndarray,
    device: torch.device,
    target_class: int | None = None,
) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for one segmentation class."""
    target_layer = dict(model.named_modules()).get("encoder.layer4")
    if target_layer is None:
        raise RuntimeError("Grad-CAM target layer encoder.layer4 was not found")

    captured = {}

    def capture_activation(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    hook = target_layer.register_forward_hook(capture_activation)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        x = torch.from_numpy(slice_chw[None]).to(device=device, dtype=torch.float32)

        with torch.enable_grad():
            logits = model(x)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                raise RuntimeError("Expected segmentation logits with shape (N,C,H,W)")

            if target_class is None:
                predicted = logits.detach().argmax(dim=1)[0]
                foreground = predicted[predicted > 0]
                target_class = (
                    int(torch.bincount(foreground).argmax().item())
                    if foreground.numel()
                    else int(logits.detach().argmax(dim=1)[0].flatten()[0].item())
                )

            if not 0 <= target_class < logits.shape[1]:
                raise ValueError(f"target_class {target_class} is outside model output classes")

            predicted = logits.detach().argmax(dim=1)
            selected_pixels = predicted == target_class
            if not selected_pixels.any():
                return np.zeros(x.shape[-2:], dtype=np.float32)

            score = (logits[:, target_class] * selected_pixels).sum()
            score.backward()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM could not capture activations and gradients")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().numpy().astype(np.float32)

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
