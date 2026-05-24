"""Frame capture sources for the pose pipeline.

`IPWebcamCapture` pulls an MJPEG stream from an Android IP Webcam app over the
LAN; it mirrors the interface of `capture.WebcamCapture` so it is a drop-in
replacement (see CLAUDE.md "streaming server for mobile client").
"""

from camera.ip_webcam import iter_jpeg_frames

__all__ = ["iter_jpeg_frames"]
