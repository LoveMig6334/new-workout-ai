"""SimCC head: backbone feature -> per-keypoint 1D logits over x and y bins.

Reduce to K channels, flatten the spatial map per keypoint, then two linear
classifiers (shared across keypoints) to the x/y bin axes.
"""
import torch
import torch.nn as nn


class SimCCHead(nn.Module):
    def __init__(self, in_channels, num_kpts, feat_h, feat_w, x_bins, y_bins):
        super().__init__()
        self.num_kpts = num_kpts
        self.reduce = nn.Conv2d(in_channels, num_kpts, 1)
        flat = feat_h * feat_w
        self.x_fc = nn.Linear(flat, x_bins)
        self.y_fc = nn.Linear(flat, y_bins)

    def forward(self, feat):
        b = feat.shape[0]
        x = self.reduce(feat)                      # (B, K, h, w)
        x = x.flatten(2)                           # (B, K, h*w)
        sx = self.x_fc(x)                          # (B, K, x_bins)
        sy = self.y_fc(x)                          # (B, K, y_bins)
        return sx, sy
