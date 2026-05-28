"""SimCC label codec: keypoints (input-space) <-> 1D Gaussian distributions.

Mirrors RTMPose's SimCC representation so teacher and student share one codec.
encode_simcc produces Gaussian-smoothed soft labels for the GT loss; decode_simcc
turns either GT targets or model logits into keypoints + a peak-value confidence.
"""
import numpy as np
from . import config

_SIGMA = 6.0  # in bin units; RTMPose-style smoothing


def _gaussian_1d(num_bins: int, mu: np.ndarray, sigma: float) -> np.ndarray:
    """(K, num_bins) Gaussian centered at mu (in bins) for each of K keypoints."""
    x = np.arange(num_bins, dtype=np.float32)[None, :]  # (1, bins)
    mu = mu[:, None]                                     # (K, 1)
    g = np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return g.astype(np.float32)


def encode_simcc(kps_input: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kps_input: (K, 2) in input space. vis: (K,) 0/1. Returns (sx (K,Xbins), sy (K,Ybins))."""
    mu_x = kps_input[:, 0] * config.SIMCC_SPLIT
    mu_y = kps_input[:, 1] * config.SIMCC_SPLIT
    sx = _gaussian_1d(config.SIMCC_X_BINS, mu_x, _SIGMA)
    sy = _gaussian_1d(config.SIMCC_Y_BINS, mu_y, _SIGMA)
    mask = (vis > 0).astype(np.float32)[:, None]
    return sx * mask, sy * mask


def decode_simcc(sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sx: (B, K, Xbins), sy: (B, K, Ybins). Returns (kps (B,K,2) input-space, scores (B,K))."""
    x_idx = sx.argmax(axis=-1).astype(np.float32)  # (B, K)
    y_idx = sy.argmax(axis=-1).astype(np.float32)
    kps = np.stack([x_idx / config.SIMCC_SPLIT, y_idx / config.SIMCC_SPLIT], axis=-1)
    # confidence = geometric-ish mean of the two peak values (softmax-free; raw peak)
    scores = np.sqrt(sx.max(axis=-1) * sy.max(axis=-1))
    return kps.astype(np.float32), scores.astype(np.float32)
