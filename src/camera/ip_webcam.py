"""MJPEG capture source for the Android IP Webcam app.

The app serves a `multipart/x-mixed-replace` MJPEG stream at `/video`. We read
the stream in a background thread, extract each complete JPEG with the pure
`iter_jpeg_frames` parser, decode it to a BGR ndarray, and keep only the latest
frame — exactly like `capture.WebcamCapture`, so it is a drop-in source.
"""

from __future__ import annotations

_SOI = b"\xff\xd8"  # JPEG Start Of Image
_EOI = b"\xff\xd9"  # JPEG End Of Image


def iter_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract every complete JPEG from `buffer`.

    Scans for SOI (0xFFD8) ... EOI (0xFFD9) marker pairs rather than parsing the
    multipart boundary string (robust to per-part header variation across IP
    Webcam app versions). Returns (complete_jpeg_blobs, remainder); `remainder`
    is the trailing partial frame (from the last unmatched SOI) or a lone 0xFF
    that may be the first half of a split SOI, to prepend to the next chunk.
    Bytes before the first SOI (e.g. multipart headers) are discarded.
    """
    frames: list[bytes] = []
    while True:
        soi = buffer.find(_SOI)
        if soi == -1:
            # No frame start. Keep a trailing 0xFF (possible split SOI), else drop.
            return frames, b"\xff" if buffer.endswith(b"\xff") else b""
        eoi = buffer.find(_EOI, soi + 2)
        if eoi == -1:
            return frames, buffer[soi:]  # incomplete frame; buffer it
        frames.append(buffer[soi : eoi + 2])
        buffer = buffer[eoi + 2 :]
