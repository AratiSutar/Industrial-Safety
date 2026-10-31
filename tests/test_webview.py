import urllib.request

import cv2
import numpy as np
import pytest

from safety.webview import WebView


@pytest.fixture
def view():
    v = WebView(port=0, host="127.0.0.1")  # port 0 = any free port
    yield v
    v.close()


def url(v, path):
    return f"http://127.0.0.1:{v.server.server_address[1]}{path}"


def test_page_and_health(view):
    assert urllib.request.urlopen(url(view, "/healthz"), timeout=5).read() == b"ok"
    assert b'<img src="/stream"' in urllib.request.urlopen(url(view, "/"), timeout=5).read()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url(view, "/nothing"), timeout=5)


def test_stream_sends_published_frame_resized(view):
    view.publish(np.full((2160, 3840, 3), 200, np.uint8))   # 4K frame
    with urllib.request.urlopen(url(view, "/stream"), timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        r.readline()                                          # --frame
        r.readline()                                          # Content-Type
        length = int(r.readline().split(b":")[1])
        r.readline()                                          # blank line
        img = cv2.imdecode(np.frombuffer(r.read(length), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 1280 and abs(int(img.mean()) - 200) < 3
