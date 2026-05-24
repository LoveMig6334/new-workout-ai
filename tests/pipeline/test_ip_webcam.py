import cv2
import numpy as np
import pytest

from camera.ip_webcam import _normalize_url, iter_jpeg_frames


def _jpeg(value: int) -> bytes:
    """A real, decodable JPEG (starts FFD8, ends FFD9) tagged by fill value."""
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_single_complete_frame():
    a = _jpeg(10)
    frames, remainder = iter_jpeg_frames(a)
    assert frames == [a]
    assert remainder == b""


def test_two_frames_in_one_buffer():
    a, b = _jpeg(10), _jpeg(200)
    frames, remainder = iter_jpeg_frames(a + b)
    assert frames == [a, b]
    assert remainder == b""


def test_frame_split_across_chunk_boundary():
    a = _jpeg(10)
    cut = len(a) // 2
    frames1, rem1 = iter_jpeg_frames(a[:cut])
    assert frames1 == []          # no EOI yet -> buffered
    assert rem1 == a[:cut]
    frames2, rem2 = iter_jpeg_frames(rem1 + a[cut:])
    assert frames2 == [a]
    assert rem2 == b""


def test_trailing_partial_frame_returned_as_remainder():
    a, b = _jpeg(10), _jpeg(200)
    partial = b[: len(b) // 2]
    frames, remainder = iter_jpeg_frames(a + partial)
    assert frames == [a]
    assert remainder == partial


def test_multipart_headers_between_frames_are_ignored():
    a, b = _jpeg(10), _jpeg(200)
    headers = b"\r\n--myboundary\r\nContent-Type: image/jpeg\r\n\r\n"
    frames, remainder = iter_jpeg_frames(a + headers + b)
    assert frames == [a, b]
    assert remainder == b""


def test_no_complete_frame_yields_nothing():
    frames, remainder = iter_jpeg_frames(b"--boundary\r\nContent-Type: image/jpeg")
    assert frames == []
    assert remainder == b""


def test_trailing_lone_ff_byte_is_kept_as_possible_split_soi():
    a = _jpeg(10)
    frames, remainder = iter_jpeg_frames(a + b"\xff")
    assert frames == [a]
    assert remainder == b"\xff"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://192.168.1.42:8080/video", "http://192.168.1.42:8080/video"),
        ("192.168.1.42:8080", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080/", "http://192.168.1.42:8080/video"),
        ("  192.168.1.42:8080  ", "http://192.168.1.42:8080/video"),
        ("http://192.168.1.42:8080/videofeed", "http://192.168.1.42:8080/videofeed"),
    ],
)
def test_normalize_url(raw, expected):
    assert _normalize_url(raw) == expected
