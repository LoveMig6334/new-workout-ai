import numpy as np
from compression import config
from compression.transforms import bbox_to_center_scale, get_warp_matrix, warp_keypoints


def test_center_scale_centers_the_bbox():
    # bbox xywh
    c, s = bbox_to_center_scale(np.array([100.0, 50.0, 40.0, 80.0]))
    assert np.allclose(c, [120.0, 90.0])  # center of the box
    assert s[0] > 0 and s[1] > 0


def test_warp_maps_bbox_center_to_input_center():
    c, s = bbox_to_center_scale(np.array([100.0, 50.0, 40.0, 80.0]))
    M = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H))
    center_in = warp_keypoints(c[None], M)[0]
    assert np.allclose(center_in, [config.INPUT_W / 2, config.INPUT_H / 2], atol=1.0)


def test_inverse_warp_round_trips():
    c, s = bbox_to_center_scale(np.array([10.0, 20.0, 200.0, 100.0]))
    M = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H))
    Minv = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H), inverse=True)
    pts = np.array([[15.0, 25.0], [120.0, 80.0]], dtype=np.float32)
    pts_in = warp_keypoints(pts, M)
    pts_back = warp_keypoints(pts_in, Minv)
    assert np.allclose(pts_back, pts, atol=1.0)
