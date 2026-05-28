"""Student = CSPNeXt-lite backbone + SimCC head, sized from config.

`input_scale` is the resolution-specialization lever: it downsamples the input
crop fed to the backbone while the SimCC bin counts stay fixed at the teacher's
(384/512), so distillation targets stay aligned and decoded keypoints remain in
the canonical 256x192 space. width/depth shrink the backbone.
"""
import torch.nn as nn
import torch.nn.functional as F
from .. import config
from .backbone import CSPNeXtLite
from .head import SimCCHead


class StudentPose(nn.Module):
    def __init__(self, width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH,
                 input_scale: float = 1.0):
        super().__init__()
        self.input_scale = input_scale
        self.arch = {"width": width, "depth": depth, "input_scale": input_scale}
        self.backbone = CSPNeXtLite(width=width, depth=depth)
        feat_h = int(config.INPUT_H * input_scale) // 32
        feat_w = int(config.INPUT_W * input_scale) // 32
        self.head = SimCCHead(
            in_channels=self.backbone.out_channels,
            num_kpts=config.NUM_KEYPOINTS,
            feat_h=feat_h, feat_w=feat_w,
            x_bins=config.SIMCC_X_BINS, y_bins=config.SIMCC_Y_BINS,
        )

    def forward(self, x):
        if self.input_scale != 1.0:
            x = F.interpolate(x, scale_factor=self.input_scale,
                              mode="bilinear", align_corners=False)
        return self.head(self.backbone(x))


def save_student(model: "StudentPose", path):
    """Checkpoint the weights AND the arch dict so any variant reloads exactly."""
    import torch
    torch.save({"state_dict": model.state_dict(), "arch": model.arch}, path)


def load_student(path, device="cpu") -> "StudentPose":
    import torch
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "arch" in blob:
        model = StudentPose(**blob["arch"])
        model.load_state_dict(blob["state_dict"])
    else:  # backward-compat: a bare state_dict
        model = StudentPose()
        model.load_state_dict(blob)
    return model.to(device).eval()
