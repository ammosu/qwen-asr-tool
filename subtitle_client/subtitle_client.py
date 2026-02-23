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


# ---------------------------------------------------------------------------
# Translation Debouncer
# ---------------------------------------------------------------------------

class TranslationDebouncer:
    """
    將英文 ASR 文字 debounce 後送 GPT-4o mini 翻譯成繁體中文。

    使用方式：
        def on_translation(zh_text):
            print(zh_text)

        debouncer = TranslationDebouncer(api_key="sk-...", callback=on_translation)
        debouncer.update("Hello world")  # 每次 ASR 更新時呼叫
        debouncer.shutdown()
    """

    SENTENCE_ENDINGS = {".", "?", "!", "。", "？", "！"}
    DEBOUNCE_SEC = 0.4

    def __init__(self, api_key: str, callback, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.callback = callback

        self._last_translated = ""
        self._pending_text = ""
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def update(self, text: str):
        """每次 ASR 更新時呼叫。text 是目前的完整轉錄文字。"""
        with self._lock:
            if text == self._pending_text:
                return
            self._pending_text = text

            # 句尾立即翻譯
            if text and text[-1] in self.SENTENCE_ENDINGS:
                self._cancel_timer()
                self._do_translate(text)
                return

            # 一般 debounce
            self._cancel_timer()
            self._timer = threading.Timer(self.DEBOUNCE_SEC, self._on_timer)
            self._timer.daemon = True
            self._timer.start()

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self):
        with self._lock:
            text = self._pending_text
        self._do_translate(text)

    def _do_translate(self, text: str):
        if not text or text == self._last_translated:
            return
        self._last_translated = text
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a real-time subtitle translator. "
                            "Translate the English speech transcript to Traditional Chinese (繁體中文). "
                            "Output ONLY the translation, no explanations."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            zh = response.choices[0].message.content.strip()
            self.callback(zh)
        except Exception as e:
            print(f"[Translation error] {e}")

    def shutdown(self):
        with self._lock:
            self._cancel_timer()
