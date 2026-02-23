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


# ---------------------------------------------------------------------------
# Subtitle Overlay Window
# ---------------------------------------------------------------------------

class SubtitleOverlay:
    """
    Always-on-top 半透明字幕視窗，固定在指定螢幕底部。

    使用方式：
        overlay = SubtitleOverlay(screen_index=0)
        overlay.set_text(en="Hello world", zh="你好世界")
        overlay.run()  # 阻塞，在主執行緒呼叫
    """

    WINDOW_HEIGHT = 120
    BG_COLOR = "#000000"
    EN_COLOR = "#888888"
    ZH_COLOR = "#ffffff"
    EN_FONT = ("Arial", 14)
    ZH_FONT = ("Microsoft JhengHei", 22, "bold")  # Windows 繁中字體

    def __init__(self, screen_index: int = 0):
        monitors = get_monitors()
        if screen_index >= len(monitors):
            screen_index = 0
        m = monitors[screen_index]
        self._x = m.x
        self._y = m.y + m.height - self.WINDOW_HEIGHT
        self._width = m.width

        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.wm_attributes("-topmost", True)
        self._root.wm_attributes("-alpha", 0.85)
        self._root.configure(bg=self.BG_COLOR)
        self._root.geometry(
            f"{self._width}x{self.WINDOW_HEIGHT}+{self._x}+{self._y}"
        )

        # 英文行
        self._en_var = tk.StringVar()
        tk.Label(
            self._root,
            textvariable=self._en_var,
            font=self.EN_FONT,
            fg=self.EN_COLOR,
            bg=self.BG_COLOR,
            anchor="w",
            padx=20,
        ).pack(fill="x", pady=(10, 0))

        # 中文行
        self._zh_var = tk.StringVar()
        tk.Label(
            self._root,
            textvariable=self._zh_var,
            font=self.ZH_FONT,
            fg=self.ZH_COLOR,
            bg=self.BG_COLOR,
            anchor="w",
            padx=20,
        ).pack(fill="x")

        # Esc 鍵退出
        self._root.bind("<Escape>", lambda e: self._root.destroy())

    def set_text(self, en: str = "", zh: str = ""):
        """從任意執行緒安全地更新字幕（用 after() 排程到主執行緒）。"""
        def _update():
            self._en_var.set(en[:120])
            self._zh_var.set(zh[:60])
        self._root.after(0, _update)

    def run(self):
        """啟動 tkinter mainloop（阻塞，必須在主執行緒呼叫）。"""
        self._root.mainloop()

# ---------------------------------------------------------------------------
# Audio Capture
# ---------------------------------------------------------------------------

TARGET_SR = 16000
CHUNK_SAMPLES = 8000  # 0.5 秒 @ 16kHz


class AudioCapture:
    """
    持續擷取麥克風音訊，每 0.5 秒呼叫一次 callback。

    使用方式：
        def on_chunk(audio: np.ndarray):
            # audio: shape (8000,), dtype float32, 16kHz
            pass

        capture = AudioCapture(callback=on_chunk)
        capture.start()
        time.sleep(10)
        capture.stop()
    """

    def __init__(self, callback: Callable[[np.ndarray], None], device=None):
        self.callback = callback
        self.device = device  # None = 系統預設麥克風
        self._stream = None

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _sd_callback(self, indata: np.ndarray, frames, time_info, status):
        if status:
            print(f"[Audio] {status}")
        chunk = indata[:, 0].copy()  # mono, shape (CHUNK_SAMPLES,)
        self.callback(chunk)

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def list_devices():
        """印出可用的音訊輸入裝置，供使用者用 --device 指定。"""
        import sounddevice as sd
        print(sd.query_devices())

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time subtitle overlay")
    parser.add_argument(
        "--asr-server",
        default="http://localhost:8000",
        help="Qwen3-ASR streaming server URL (e.g. http://192.168.1.100:8000)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=0,
        help="Display screen index (0=primary, 1=secondary)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (run with --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--translation-model",
        default="gpt-4o-mini",
        help="OpenAI model for translation",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioCapture.list_devices()
        return

    if not args.openai_api_key:
        print("Error: --openai-api-key or OPENAI_API_KEY required")
        return

    # 初始化字幕視窗
    overlay = SubtitleOverlay(screen_index=args.screen)

    # 目前的 ASR 文字（跨執行緒共享）
    current_en = ""

    def on_translation(zh_text: str):
        nonlocal current_en
        overlay.set_text(en=current_en, zh=zh_text)

    # 初始化翻譯 debouncer
    debouncer = TranslationDebouncer(
        api_key=args.openai_api_key,
        callback=on_translation,
        model=args.translation_model,
    )

    # 初始化 ASR client
    asr = ASRClient(args.asr_server)

    def run_asr():
        nonlocal current_en
        asr.start()
        print(f"[ASR] Session started: {asr.session_id}")

        def on_chunk(audio: np.ndarray):
            nonlocal current_en
            try:
                result = asr.push_chunk(audio)
                text = result.get("text", "")
                if text != current_en:
                    current_en = text
                    overlay.set_text(en=text, zh="")
                    debouncer.update(text)
            except Exception as e:
                print(f"[ASR error] {e}")

        capture = AudioCapture(callback=on_chunk, device=args.device)
        capture.start()
        print("[Audio] Capturing... Press Esc to stop.")

        try:
            while overlay._root.winfo_exists():
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            capture.stop()
            debouncer.shutdown()
            try:
                asr.finish()
            except Exception:
                pass
            print("[ASR] Session finished.")

    # ASR 在背景執行緒跑
    asr_thread = threading.Thread(target=run_asr, daemon=True)
    asr_thread.start()

    # tkinter mainloop 在主執行緒（blocking）
    overlay.run()


if __name__ == "__main__":
    main()
