import numpy as np
from compression.softlabel_store import SoftLabelWriter, SoftLabelReader


def test_write_then_read_round_trips_fp16(tmp_path):
    w = SoftLabelWriter(str(tmp_path / "sl"), x_bins=384, y_bins=512, num_kpts=17)
    sx = np.random.rand(17, 384).astype(np.float32)
    sy = np.random.rand(17, 512).astype(np.float32)
    w.add(ann_id=42, sx=sx, sy=sy)
    w.close()

    r = SoftLabelReader(str(tmp_path / "sl"))
    assert 42 in r.ann_ids()
    rx, ry = r.get(42)
    # stored as fp16 -> compare with fp16 tolerance
    assert np.allclose(rx, sx, atol=1e-2)
    assert np.allclose(ry, sy, atol=1e-2)
    assert rx.dtype == np.float16
