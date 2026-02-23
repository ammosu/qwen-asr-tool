#!/usr/bin/env python3
"""
Real-time subtitle overlay for Windows.

Usage:
    python subtitle_client.py --asr-server http://<LINUX_IP>:8000 --openai-api-key sk-...

Requirements:
    pip install sounddevice numpy requests openai screeninfo
"""
import argparse
import os
import threading
import time
from typing import Callable

import numpy as np
import requests
import tkinter as tk
from openai import OpenAI
from screeninfo import get_monitors


# ---------------------------------------------------------------------------
# ASR Client
# ---------------------------------------------------------------------------

class ASRClient:
    """HTTP client for Qwen3-ASR streaming server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    def start(self):
        """建立新的 streaming session。"""
        r = requests.post(f"{self.base_url}/api/start", timeout=10)
        r.raise_for_status()
        self.session_id = r.json()["session_id"]

    def push_chunk(self, audio_float32: np.ndarray) -> dict:
        """
        送出一段 16kHz float32 音訊，回傳 {"language": str, "text": str}。
        audio_float32: shape (N,), dtype float32
        """
        assert self.session_id, "Call start() first"
        r = requests.post(
            f"{self.base_url}/api/chunk",
            params={"session_id": self.session_id},
            data=audio_float32.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def finish(self) -> dict:
        """結束 session，回傳最終結果。"""
        assert self.session_id, "Call start() first"
        r = requests.post(
            f"{self.base_url}/api/finish",
            params={"session_id": self.session_id},
            timeout=10,
        )
        r.raise_for_status()
        self.session_id = None
        return r.json()
