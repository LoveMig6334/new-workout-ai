import numpy as np

from analysis.angles_3d import body_frame_axes


def _canonical_standing_pose() -> np.ndarray:
    """Pelvis at origin, thorax 1 unit up (image-y down → up = -y),
    hips along ±x. Other H36M joints zero-initialized (unused by this fn)."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)       # pelvis
    kps[1] = (0.2, 0.0, 0.0)       # R_hip
    kps[4] = (-0.2, 0.0, 0.0)      # L_hip
    kps[8] = (0.0, -1.0, 0.0)      # thorax
    return kps


def test_body_frame_axes_canonical_pose_is_orthonormal():
    kps = _canonical_standing_pose()
    up, lat, fwd = body_frame_axes(kps)
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(lat, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(fwd, [0.0, 0.0, 1.0], atol=1e-6)


def test_body_frame_axes_are_unit_vectors():
    kps = _canonical_standing_pose()
    for v in body_frame_axes(kps):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_body_frame_axes_lateral_is_orthogonal_to_up_even_when_hipline_tilted():
    kps = _canonical_standing_pose()
    # Tilt the hip line so it isn't already perpendicular to body_up.
    kps[1] = (0.2, -0.1, 0.0)
    kps[4] = (-0.2, 0.1, 0.0)
    up, lat, _ = body_frame_axes(kps)
    assert abs(np.dot(up, lat)) < 1e-6
