"""Browser preview: serves the latest annotated, face-blurred frame as an MJPEG stream.

Used instead of the preview window where there is no screen (e.g. inside Docker).
Built on the standard library only. Raw camera frames are never served.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial Safety</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}
img{display:block;width:100%;height:auto}p{margin:8px 12px;font-size:14px}</style></head>
<body><img src="/stream" alt="Live view (faces blurred)">
<p>Live view, faces blurred. Operator buttons arrive with the operator screen.</p></body></html>"""


class WebView:
    def __init__(self, port: int, host: str = "0.0.0.0", max_width: int = 1280, quality: int = 75):
        self.max_width = max_width
        self.quality = quality
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="webview").start()

    def publish(self, frame: np.ndarray) -> None:
        """Share a frame that is already annotated and face-blurred."""
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()

    def _next(self, last_seq: int) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=5.0)
            return self._seq, self._jpeg

    def _handler(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal for alerts
                pass

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    self._send(200, PAGE, "text/html; charset=utf-8")
                elif self.path == "/healthz":
                    self._send(200, b"ok", "text/plain")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seq = -1
                    try:
                        while True:
                            seq, jpeg = view._next(seq)
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        return  # browser tab closed
                else:
                    self._send(404, b"not found", "text/plain")

        return Handler
