"""Central config for the RTMPose compression project.

INPUT_H/W and the SIMCC bin counts are the RTMPose-m defaults; they are
re-verified against the teacher's real ONNX output shapes in
export_softlabels.py (which asserts on mismatch) so they can never drift
silently.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Top-down model input (H, W) and SimCC split ratio (RTMPose-m defaults).
INPUT_H = 256
INPUT_W = 192
SIMCC_SPLIT = 2.0
SIMCC_X_BINS = int(INPUT_W * SIMCC_SPLIT)  # 384
SIMCC_Y_BINS = int(INPUT_H * SIMCC_SPLIT)  # 512
NUM_KEYPOINTS = 17

# Top-down crop padding around the bbox (matches rtmlib's RTMPose default).
BBOX_PADDING = 1.25

# ImageNet-style normalization used by RTMPose.
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)

# Where artifacts land (all gitignored under models/ or a sibling).
SOFTLABEL_DIR = MODELS_DIR / "compression" / "softlabels"
CHECKPOINT_DIR = MODELS_DIR / "compression" / "checkpoints"
COCO_ROOT = PROJECT_ROOT / "data" / "coco"  # images + annotations live here

# Student defaults (overridable on the train.py CLI).
STUDENT_WIDTH = 0.33   # backbone channel multiplier vs the reference
STUDENT_DEPTH = 0.33   # backbone block-count multiplier
