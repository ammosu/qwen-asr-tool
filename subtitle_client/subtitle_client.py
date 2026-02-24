#!/usr/bin/env python3
"""
Real-time subtitle overlay（Linux/Windows）。

Usage:
    python subtitle_client.py --asr-server http://<SERVER_IP>:8000 --openai-api-key sk-...

Requirements:
    pip install sounddevice numpy scipy requests openai screeninfo
"""
# 必須在所有 X11/tkinter/sounddevice import 之前呼叫，
# 避免 PulseAudio PortAudio 後端的 XInitThreads() 與 tkinter 衝突。
try:
    import ctypes
    ctypes.CDLL("libX11.so.6").XInitThreads()
except Exception:
    pass  # Windows 或找不到 libX11 時忽略

import argparse
import os
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import requests
import scipy.signal as signal
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
        self.direction: str = "en→zh"   # 目前翻譯方向

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

    def toggle_direction(self) -> str:
        """切換翻譯方向，回傳新方向字串。"""
        with self._lock:
            self.direction = "zh→en" if self.direction == "en→zh" else "en→zh"
            self._last_translated = ""  # 清空快取，強制重新翻譯
            return self.direction

    def set_direction(self, direction: str) -> None:
        """直接設定方向（'en→zh' 或 'zh→en'）。"""
        with self._lock:
            self.direction = direction
            self._last_translated = ""

    def _do_translate(self, text: str):
        with self._lock:
            if not text or text == self._last_translated:
                return
            self._last_translated = text
            direction = self.direction  # snapshot
        # lock 釋放後才呼叫 OpenAI
        if direction == "en→zh":
            system_msg = (
                "You are a real-time subtitle translator. "
                "Translate the English speech transcript to Traditional Chinese (繁體中文台灣用語). "
                "Output ONLY the translation, no explanations."
            )
        else:  # zh→en
            system_msg = (
                "You are a real-time subtitle translator. "
                "Translate the Chinese speech transcript to English. "
                "Output ONLY the translation, no explanations."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            translated = response.choices[0].message.content.strip()
            self.callback(translated)
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
        overlay.set_text(original="Hello world", translated="你好世界")
        overlay.run()  # 阻塞，在主執行緒呼叫
    """

    TOOLBAR_HEIGHT = 28
    WINDOW_HEIGHT = 148         # 原 120 + TOOLBAR_HEIGHT
    TOOLBAR_BG = "#1a1a1a"
    BTN_COLOR = "#ffffff"
    BTN_BG = "#333333"
    BG_COLOR = "#000000"
    EN_COLOR = "#888888"
    ZH_COLOR = "#ffffff"
    EN_FONT = ("Arial", 14)
    ZH_FONT = ("Microsoft JhengHei", 22, "bold")  # Windows 繁中字體

    def __init__(self, screen_index: int = 0, on_toggle_direction=None):
        self._on_toggle_direction = on_toggle_direction

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

        # ── 工具列 ──
        toolbar = tk.Frame(self._root, bg=self.TOOLBAR_BG, height=self.TOOLBAR_HEIGHT)
        toolbar.pack(fill="x", side="top")

        self._dir_btn_var = tk.StringVar(value="[EN→ZH ⇄]")
        tk.Button(
            toolbar,
            textvariable=self._dir_btn_var,
            font=("Arial", 10),
            fg=self.BTN_COLOR,
            bg=self.BTN_BG,
            relief="flat",
            padx=8,
            command=self._toggle_direction,
        ).pack(side="left", padx=4, pady=2)

        tk.Button(
            toolbar,
            text="✕",
            font=("Arial", 10),
            fg=self.BTN_COLOR,
            bg=self.BTN_BG,
            relief="flat",
            padx=8,
            command=self._root.destroy,
        ).pack(side="right", padx=4, pady=2)

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
        self._root.bind("<F9>", lambda e: self._toggle_direction())

    def _toggle_direction(self):
        if self._on_toggle_direction:
            new_dir = self._on_toggle_direction()
            self.update_direction_label(new_dir)

    def update_direction_label(self, direction: str):
        label = f"[{direction} ⇄]"
        self._root.after(0, lambda: self._dir_btn_var.set(label))

    def set_text(self, original: str = "", translated: str = ""):
        """從任意執行緒安全地更新字幕（用 after() 排程到主執行緒）。"""
        def _update():
            self._en_var.set(original[:120])
            self._zh_var.set(translated[:60])
        self._root.after(0, _update)

    def run(self):
        """啟動 tkinter mainloop（阻塞，必須在主執行緒呼叫）。"""
        self._root.mainloop()

# ---------------------------------------------------------------------------
# Audio Sources
# ---------------------------------------------------------------------------

TARGET_SR = 16000
CHUNK_SAMPLES = 8000  # 0.5 秒 @ 16kHz


class AudioSource(ABC):
    """音訊來源抽象介面。未來可新增 MicrophoneAudioSource、NetworkAudioSource 等。"""

    @abstractmethod
    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """開始擷取音訊，每 0.5 秒以 16kHz float32 mono ndarray 呼叫 callback。"""

    @abstractmethod
    def stop(self) -> None:
        """停止擷取。"""

    @staticmethod
    def list_devices() -> None:
        """列出系統音訊裝置及 PulseAudio monitor sources。"""
        import sounddevice as sd
        print("=== ALSA 裝置清單 ===")
        print(sd.query_devices())
        print("\n=== PulseAudio Monitor Sources（可用於 --monitor-device）===")
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=3,
            )
            for line in result.stdout.splitlines():
                if "monitor" in line.lower():
                    print(" ", line)
        except Exception:
            print("  （無法取得 PulseAudio sources，請確認 pactl 已安裝）")


class MonitorAudioSource(AudioSource):
    """
    擷取 PipeWire/PulseAudio monitor source（系統播放音訊）。

    使用 queue.Queue 解耦音訊 callback 與 ASR HTTP 請求，避免
    阻塞操作污染即時音訊執行緒。

    透過 ALSA pulse 設備 + PULSE_SOURCE 環境變數選擇 monitor source，
    讓 sounddevice 能存取 PipeWire/PulseAudio monitor。

    device 預設：alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor
    """

    DEFAULT_DEVICE = "alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor"
    ALSA_PULSE_DEVICE = "pulse"  # ALSA pulse plugin，透過它存取 PulseAudio

    def __init__(self, device: str | None = None):
        self._device = device or self.DEFAULT_DEVICE  # PulseAudio source 名稱
        self._stream = None
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._native_sr: int = 0
        self._callback: Callable[[np.ndarray], None] | None = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._consumer_thread: threading.Thread | None = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._stream is not None:
            raise RuntimeError("MonitorAudioSource is already running; call stop() first.")

        import sounddevice as sd

        # 設定 PULSE_SOURCE 讓 PulseAudio 使用指定的 monitor source
        os.environ["PULSE_SOURCE"] = self._device

        # 透過 ALSA pulse 設備取得 native samplerate
        dev_info = sd.query_devices(self.ALSA_PULSE_DEVICE, kind="input")
        self._native_sr = int(dev_info["default_samplerate"])  # 通常 44100 或 48000
        self._callback = callback
        self._buf = np.zeros(0, dtype=np.float32)
        self._running = True

        # 消費者執行緒：從 queue 取音訊、resample、送 callback
        self._consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self._consumer_thread.start()

        # 音訊 stream：callback 只做 enqueue（不阻塞）
        self._stream = sd.InputStream(
            samplerate=self._native_sr,
            channels=1,
            dtype="float32",
            blocksize=int(self._native_sr * 0.05),  # 50ms 固定 buffer
            device=self.ALSA_PULSE_DEVICE,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _sd_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """音訊執行緒 callback：只做最輕量的 enqueue，不做任何阻塞操作。"""
        if status:
            print(f"[Audio] {status}")
        self._queue.put(indata[:, 0].copy())

    def _consumer(self) -> None:
        """消費者執行緒：resample + 累積 buffer + 呼叫 ASR callback。"""
        while self._running:
            try:
                raw = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # resample native_sr → 16kHz（在非即時執行緒中進行）
            target_len = int(len(raw) * TARGET_SR / self._native_sr)
            resampled = signal.resample(raw, target_len).astype(np.float32)
            self._buf = np.concatenate([self._buf, resampled])

            # 每累積 CHUNK_SAMPLES 就送出一次
            while len(self._buf) >= CHUNK_SAMPLES:
                chunk = self._buf[:CHUNK_SAMPLES].copy()
                self._buf = self._buf[CHUNK_SAMPLES:]
                if self._callback:
                    self._callback(chunk)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._consumer_thread:
            self._consumer_thread.join(timeout=1.0)
            self._consumer_thread = None
        self._buf = np.zeros(0, dtype=np.float32)


class MicrophoneAudioSource(AudioSource):
    """麥克風音訊來源（預留，尚未實作）。"""

    def __init__(self, device=None):
        self._device = device

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        raise NotImplementedError("MicrophoneAudioSource 尚未實作")

    def stop(self) -> None:
        pass

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
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--translation-model",
        default="gpt-4o-mini",
        help="OpenAI model for translation",
    )
    parser.add_argument(
        "--source",
        choices=["monitor", "mic"],
        default="monitor",
        help="Audio source type",
    )
    parser.add_argument(
        "--monitor-device",
        default=MonitorAudioSource.DEFAULT_DEVICE,
        help="PipeWire/PulseAudio monitor device name",
    )
    parser.add_argument(
        "--direction",
        choices=["en→zh", "zh→en"],
        default="en→zh",
        help="Translation direction",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioSource.list_devices()
        return

    if not args.openai_api_key:
        print("Error: --openai-api-key or OPENAI_API_KEY required")
        return

    # 建立翻譯 debouncer，callback 先設成佔位 lambda
    debouncer = TranslationDebouncer(
        api_key=args.openai_api_key,
        callback=lambda t: None,
        model=args.translation_model,
    )
    debouncer.set_direction(args.direction)

    # 建立字幕視窗
    overlay = SubtitleOverlay(
        screen_index=args.screen,
        on_toggle_direction=debouncer.toggle_direction,
    )
    overlay.update_direction_label(args.direction)

    # 目前的 ASR 文字（跨執行緒共享）
    current_original = ""

    def on_translation(translated: str):
        nonlocal current_original
        overlay.set_text(original=current_original, translated=translated)

    debouncer.callback = on_translation

    # 根據 args.source 選擇音訊來源
    if args.source == "monitor":
        audio_source = MonitorAudioSource(device=args.monitor_device)
    else:
        audio_source = MicrophoneAudioSource()

    # 初始化 ASR client
    asr = ASRClient(args.asr_server)

    def run_asr():
        nonlocal current_original
        asr.start()
        print(f"[ASR] Session started: {asr.session_id}")

        def on_chunk(audio: np.ndarray):
            nonlocal current_original
            try:
                result = asr.push_chunk(audio)
                text = result.get("text", "")
                if text != current_original:
                    current_original = text
                    overlay.set_text(original=text, translated="")
                    debouncer.update(text)
            except Exception as e:
                print(f"[ASR error] {e}")

        audio_source.start(on_chunk)
        print("[Audio] Capturing... Press Esc to stop.")

        try:
            while overlay._root.winfo_exists():
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            audio_source.stop()
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
