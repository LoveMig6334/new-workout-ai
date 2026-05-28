import numpy as np
from compression import config
from compression.simcc import encode_simcc, decode_simcc


def test_encode_shapes_and_peak_at_keypoint():
    # one keypoint at input-space (x=96, y=128) -> peak bin = coord * split
    kps = np.array([[96.0, 128.0]], dtype=np.float32)  # (K=1, 2)
    vis = np.array([1.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    assert sx.shape == (1, config.SIMCC_X_BINS)
    assert sy.shape == (1, config.SIMCC_Y_BINS)
    assert int(sx[0].argmax()) == round(96.0 * config.SIMCC_SPLIT)
    assert int(sy[0].argmax()) == round(128.0 * config.SIMCC_SPLIT)


def test_invisible_keypoint_is_zero_target():
    kps = np.array([[96.0, 128.0]], dtype=np.float32)
    vis = np.array([0.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    assert np.allclose(sx, 0.0) and np.allclose(sy, 0.0)


def test_decode_round_trips_encode():
    kps = np.array([[40.0, 200.0], [150.0, 50.0]], dtype=np.float32)
    vis = np.array([1.0, 1.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    out_kps, scores = decode_simcc(sx[None], sy[None])  # add batch dim
    assert out_kps.shape == (1, 2, 2)
    # within half a bin (1 / split) of the input
    assert np.allclose(out_kps[0], kps, atol=1.0)
    assert np.all(scores[0] > 0.5)
