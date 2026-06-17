#!/usr/bin/env python3
from __future__ import annotations

import socket
import sys
import urllib.request


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def tcp_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


if not http_ok("http://127.0.0.1:6080/vnc.html"):
    sys.exit(1)

if not tcp_ok("127.0.0.1", 5900):
    sys.exit(1)

sys.exit(0)
