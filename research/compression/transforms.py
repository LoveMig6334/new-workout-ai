"""Top-down affine preprocessing matching rtmlib's RTMPose convention:
bbox -> center/scale (with BBOX_PADDING, aspect fixed to the input ratio) ->
affine warp to (INPUT_W, INPUT_H). Used identically by teacher export and the
student dataset so their SimCC outputs are aligned.
"""
import cv2
import numpy as np
from . import config


def bbox_to_center_scale(bbox_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = bbox_xywh
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    aspect = config.INPUT_W / config.INPUT_H
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    scale = np.array([w, h], dtype=np.float32) * config.BBOX_PADDING
    return center, scale


def get_warp_matrix(center, scale, output_size, inverse: bool = False) -> np.ndarray:
    """2x3 affine. output_size = (W, H). Maps image space -> input space (or inverse)."""
    out_w, out_h = output_size
    src_w, src_h = float(scale[0]), float(scale[1])
    src = np.array([
        center,
        center + np.array([0.0, -src_h * 0.5], dtype=np.float32),
        center + np.array([-src_w * 0.5, 0.0], dtype=np.float32),
    ], dtype=np.float32)
    dst = np.array([
        [out_w * 0.5, out_h * 0.5],
        [out_w * 0.5, out_h * 0.5 - out_h * 0.5],
        [out_w * 0.5 - out_w * 0.5, out_h * 0.5],
    ], dtype=np.float32)
    if inverse:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def warp_keypoints(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine M to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)  # (N, 3)
    return (homo @ M.T).astype(np.float32)


def warp_image(image_bgr: np.ndarray, center, scale) -> np.ndarray:
    """Warp + normalize -> (3, H, W) float32 tensor (CHW, RGB, normalized)."""
    M = get_warp_matrix(center, scale, (config.INPUT_W, config.INPUT_H))
    crop = cv2.warpAffine(image_bgr, M, (config.INPUT_W, config.INPUT_H), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - np.array(config.PIXEL_MEAN)) / np.array(config.PIXEL_STD)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)
