"""On-disk fp16 store for teacher SimCC soft labels, keyed by COCO ann_id.

Layout: a single .npz-per-shard would reload everything; instead use a flat
memmap of fixed-size records + a JSON index of ann_id -> row. fp16 halves the
~30 KB/instance footprint.
"""
import json
import numpy as np
from pathlib import Path


class SoftLabelWriter:
    def __init__(self, prefix: str, x_bins: int, y_bins: int, num_kpts: int):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.x_bins, self.y_bins, self.num_kpts = x_bins, y_bins, num_kpts
        self.row_floats = num_kpts * (x_bins + y_bins)
        self._rows = []           # list of fp16 arrays
        self._index = {}          # ann_id -> row idx

    def add(self, ann_id: int, sx: np.ndarray, sy: np.ndarray):
        rec = np.concatenate([sx.reshape(-1), sy.reshape(-1)]).astype(np.float16)
        assert rec.size == self.row_floats
        self._index[int(ann_id)] = len(self._rows)
        self._rows.append(rec)

    def close(self):
        arr = np.stack(self._rows, axis=0) if self._rows else \
            np.zeros((0, self.row_floats), np.float16)
        np.save(str(self.prefix) + ".npy", arr)
        meta = {"x_bins": self.x_bins, "y_bins": self.y_bins,
                "num_kpts": self.num_kpts, "index": self._index}
        Path(str(self.prefix) + ".json").write_text(json.dumps(meta))


class SoftLabelReader:
    def __init__(self, prefix: str):
        meta = json.loads(Path(str(prefix) + ".json").read_text())
        self.x_bins, self.y_bins, self.num_kpts = meta["x_bins"], meta["y_bins"], meta["num_kpts"]
        self._index = {int(k): v for k, v in meta["index"].items()}
        self._arr = np.load(str(prefix) + ".npy", mmap_mode="r")

    def ann_ids(self):
        return set(self._index.keys())

    def get(self, ann_id: int):
        row = self._arr[self._index[int(ann_id)]]
        nx = self.num_kpts * self.x_bins
        sx = np.asarray(row[:nx]).reshape(self.num_kpts, self.x_bins)
        sy = np.asarray(row[nx:]).reshape(self.num_kpts, self.y_bins)
        return sx, sy
