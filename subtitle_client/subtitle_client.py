#!/usr/bin/env python3
"""
Real-time subtitle overlay（Linux/Windows）。

Usage:
    python subtitle_client.py --asr-server http://<SERVER_IP>:8000 --openai-api-key sk-...

Requirements:
    pip install sounddevice numpy scipy requests openai
"""
import argparse
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import requests
import scipy.signal as signal
import tkinter as tk
from openai import OpenAI


# ---------------------------------------------------------------------------
# ASR Client
# ---------------------------------------------------------------------------

class ASRClient:
    """HTTP client for Qwen3-ASR server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def transcribe(self, audio_float32: np.ndarray) -> dict:
        """
        One-shot 轉錄：送出整段 16kHz float32 音訊，回傳 {"language": str, "text": str}。
        audio_float32: shape (N,), dtype float32
        """
        r = requests.post(
            f"{self.base_url}/api/transcribe",
            data=audio_float32.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=20,
        )
        r.raise_for_status()
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
        translate_now = None
        with self._lock:
            if text == self._pending_text:
                return
            self._pending_text = text

            # 句尾立即翻譯（注意：_do_translate 必須在 lock 釋放後呼叫）
            if text and text[-1] in self.SENTENCE_ENDINGS:
                self._cancel_timer()
                translate_now = text
            else:
                # 一般 debounce
                self._cancel_timer()
                self._timer = threading.Timer(self.DEBOUNCE_SEC, self._on_timer)
                self._timer.daemon = True
                self._timer.start()

        # lock 已釋放，才可呼叫 OpenAI（否則 _do_translate 內的 with self._lock 會死鎖）
        if translate_now:
            self._do_translate(translate_now)

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
                "你是即時字幕翻譯員。將英文語音轉錄翻譯成自然流暢的繁體中文（台灣口語用語）。"
                "要求：\n"
                "1. 依照中文語法重新組句，不要逐字翻譯或照搬英文語序\n"
                "2. 使用台灣人日常說話的方式，口語自然\n"
                "3. 專有名詞、人名、品牌可保留英文原文\n"
                "4. 只輸出翻譯結果，不加任何解釋或標注"
            )
        else:  # zh→en
            system_msg = (
                "You are a real-time subtitle translator. "
                "Translate the Chinese speech transcript to natural, colloquial English. "
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
            print(f"[Translation] {translated}", flush=True)
            self.callback(translated)
        except Exception as e:
            print(f"[Translation error] {e}", flush=True)

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
    DRAG_BAR_HEIGHT = 14
    WINDOW_HEIGHT = 160          # DRAG_BAR_HEIGHT + 146 (字幕區)
    WINDOW_WIDTH = 900           # 預設值，__init__ 會依螢幕動態覆蓋
    RESIZE_SIZE = 28
    TOOLBAR_BG = "#222222"
    DRAG_BAR_COLOR = "#2a2a2a"   # 深灰，非純黑（不會被 transparentcolor 穿透）
    BTN_COLOR = "#ffffff"
    BTN_BG = "#333333"
    BG_COLOR = "#000000"
    EN_COLOR = "#dddddd"
    ZH_COLOR = "#ffffff"
    SHADOW_COLOR = "#111111"
    EN_FONT = ("Arial", 15)
    ZH_FONT = ("Microsoft JhengHei", 22, "bold")  # Windows 繁中字體

    def __init__(self, screen_index: int = 0, on_toggle_direction=None, on_switch_source=None):
        self._on_toggle_direction = on_toggle_direction
        self._on_switch_source = on_switch_source

        self._root = tk.Tk()

        # 用 tkinter 取螢幕尺寸（不依賴 screeninfo）
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        # 視窗寬度為螢幕的 80%（最小 900），高度固定
        self._win_w = max(900, int(screen_w * 0.80))
        self._win_h = self.WINDOW_HEIGHT
        self._x = (screen_w - self._win_w) // 2
        self._y = screen_h - self._win_h - 40

        self._root.overrideredirect(True)
        self._root.wm_attributes("-topmost", True)
        if sys.platform == "win32":
            self._root.wm_attributes("-transparentcolor", self.BG_COLOR)
        else:
            self._root.wm_attributes("-alpha", 0.85)
        self._root.configure(bg=self.BG_COLOR)
        self._root.geometry(
            f"{self._win_w}x{self._win_h}+{self._x}+{self._y}"
        )

        # ── 拖拉條（常駐頂部，提供拖拉控點） ──
        drag_bar = tk.Frame(
            self._root,
            bg=self.DRAG_BAR_COLOR,
            height=self.DRAG_BAR_HEIGHT,
            cursor="fleur",          # 十字箭頭游標，提示可拖拉
        )
        drag_bar.pack(fill="x", side="top")
        drag_bar.pack_propagate(False)
        # 拖拉綁定在拖拉條上，不影響字幕區
        drag_bar.bind("<ButtonPress-1>", self._start_drag)
        drag_bar.bind("<B1-Motion>", self._do_drag)

        # ── Canvas (created after drag bar, fills remaining space) ──
        self._canvas = tk.Canvas(
            self._root,
            bg=self.BG_COLOR,
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda e: self._redraw_text())

        # 按下時記錄起始狀態，motion/release 改綁到 root（拖出三角形後仍持續追蹤）
        self._canvas.tag_bind("resize_handle", "<ButtonPress-1>", self._start_resize)

        # ── 工具列 (created after canvas so it has higher z-order) ──
        toolbar = tk.Frame(self._root, bg=self.TOOLBAR_BG, height=self.TOOLBAR_HEIGHT)
        toolbar.place(x=0, y=0, relwidth=1.0, height=self.TOOLBAR_HEIGHT)
        toolbar.place_forget()
        self._toolbar = toolbar

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

        self._src_btn_var = tk.StringVar(value="[🔊 MON]")
        tk.Button(
            toolbar,
            textvariable=self._src_btn_var,
            font=("Arial", 10),
            fg=self.BTN_COLOR,
            bg=self.BTN_BG,
            relief="flat",
            padx=8,
            command=self._switch_source,
        ).pack(side="left", padx=4, pady=2)

        tk.Button(
            toolbar,
            text="✕",
            font=("Arial", 10),
            fg=self.BTN_COLOR,
            bg=self.BTN_BG,
            relief="flat",
            padx=8,
            command=self._do_close,
        ).pack(side="right", padx=4, pady=2)

        self._toolbar_hide_id = None

        # 工具列由拖拉條觸發（hover 拖拉條 → 工具列展開並覆蓋拖拉條）
        # 工具列本身也支援拖拉（按住工具列空白處拖動）
        drag_bar.bind("<Enter>", self._show_toolbar)
        drag_bar.bind("<Leave>", self._hide_toolbar)
        self._toolbar.bind("<Enter>", self._show_toolbar)
        self._toolbar.bind("<Leave>", self._hide_toolbar)
        self._toolbar.bind("<ButtonPress-1>", self._start_drag)
        self._toolbar.bind("<B1-Motion>", self._do_drag)

        self._en_str = ""
        self._zh_str = ""
        self._drag_x = 0
        self._drag_y = 0
        self._resize_start = None   # (mouse_x, mouse_y, win_w, win_h)

        self._root.bind("<Escape>", lambda e: self._do_close())
        self._root.bind("<F9>", lambda e: self._toggle_direction())
        self._root.protocol("WM_DELETE_WINDOW", self._do_close)

    def _do_close(self):
        """關閉視窗。"""
        self._root.destroy()

    def _show_toolbar(self, event=None):
        if self._toolbar_hide_id:
            self._root.after_cancel(self._toolbar_hide_id)
            self._toolbar_hide_id = None
        self._toolbar.place(x=0, y=0, relwidth=1.0, height=self.TOOLBAR_HEIGHT)
        self._toolbar.lift()

    def _hide_toolbar(self, event=None):
        self._toolbar_hide_id = self._root.after(
            400, lambda: self._toolbar.place_forget()
        )

    def _start_drag(self, event):
        self._drag_x = event.x_root - self._root.winfo_x()
        self._drag_y = event.y_root - self._root.winfo_y()

    def _do_drag(self, event):
        nx = event.x_root - self._drag_x
        ny = event.y_root - self._drag_y
        self._root.geometry(f"+{nx}+{ny}")

    def _draw_resize_handle(self):
        """Draw a small triangle at bottom-right of canvas for resizing."""
        self._canvas.delete("resize_handle")
        w = self._canvas.winfo_width() or self._root.winfo_width()
        h = self._canvas.winfo_height() or self._root.winfo_height()
        s = self.RESIZE_SIZE
        self._canvas.create_polygon(
            w, h - s,
            w - s, h,
            w, h,
            fill="#aaaaaa", outline="", tags="resize_handle",
        )
        self._canvas.tag_bind("resize_handle", "<Enter>",
                              lambda e: self._canvas.configure(cursor="sizing"))
        self._canvas.tag_bind("resize_handle", "<Leave>",
                              lambda e: self._canvas.configure(cursor=""))

    def _start_resize(self, event):
        self._resize_start = (
            event.x_root, event.y_root,
            self._root.winfo_width(), self._root.winfo_height(),
        )
        # 綁到 root，拖出三角形範圍後仍可持續縮放
        self._root.bind("<B1-Motion>",       self._do_resize)
        self._root.bind("<ButtonRelease-1>", self._stop_resize)
        return "break"

    def _do_resize(self, event):
        if not self._resize_start:
            return
        mx0, my0, w0, h0 = self._resize_start
        new_w = max(300, w0 + event.x_root - mx0)
        new_h = max(80,  h0 + event.y_root - my0)
        x = self._root.winfo_x()
        y = self._root.winfo_y()
        self._root.geometry(f"{new_w}x{new_h}+{x}+{y}")

    def _stop_resize(self, event):
        self._resize_start = None
        self._root.unbind("<B1-Motion>")
        self._root.unbind("<ButtonRelease-1>")

    def _toggle_direction(self):
        if self._on_toggle_direction:
            new_dir = self._on_toggle_direction()
            self.update_direction_label(new_dir)

    def update_direction_label(self, direction: str):
        label = f"[{direction} ⇄]"
        self._root.after(0, lambda: self._dir_btn_var.set(label))

    def _switch_source(self):
        if self._on_switch_source:
            self._on_switch_source()

    def update_source_label(self, source: str):
        label = "[🎤 MIC]" if source == "mic" else "[🔊 MON]"
        self._root.after(0, lambda: self._src_btn_var.set(label))

    def set_text(self, original: str = "", translated: str = ""):
        """從任意執行緒安全地更新字幕（用 after() 排程到主執行緒）。"""
        def _update():
            self._en_str = original[-120:] if len(original) > 120 else original
            self._zh_str = translated[-60:] if len(translated) > 60 else translated
            self._redraw_text()
        self._root.after(0, _update)

    def _redraw_text(self):
        """Clear canvas and re-draw subtitle text with shadow."""
        self._canvas.delete("text")

        w = self._canvas.winfo_width() or self._root.winfo_width()
        wrap_w = max(200, w - 40)   # ensure positive wrap width

        # EN line — 20px from left, 12px from top of canvas area
        ex, ey = 20, 12
        self._canvas.create_text(ex+2, ey+2, text=self._en_str, fill=self.SHADOW_COLOR,
                                 font=self.EN_FONT, anchor="nw", width=wrap_w, tags="text")
        self._canvas.create_text(ex,   ey,   text=self._en_str, fill=self.EN_COLOR,
                                 font=self.EN_FONT, anchor="nw", width=wrap_w, tags="text")

        # ZH line — below EN (~30px gap covers Arial-15 line height)
        zy = ey + 30
        self._canvas.create_text(ex+2, zy+2, text=self._zh_str, fill=self.SHADOW_COLOR,
                                 font=self.ZH_FONT, anchor="nw", width=wrap_w, tags="text")
        self._canvas.create_text(ex,   zy,   text=self._zh_str, fill=self.ZH_COLOR,
                                 font=self.ZH_FONT, anchor="nw", width=wrap_w, tags="text")

        self._draw_resize_handle()

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
        """列出系統音訊裝置。"""
        import sounddevice as sd
        print("=== 音訊裝置清單 ===")
        print(sd.query_devices())
        if sys.platform == "win32":
            print("\n=== WASAPI Loopback 可用裝置（可用於 --monitor-device）===")
            try:
                wasapi_idx = next(
                    (i for i, api in enumerate(sd.query_hostapis()) if "wasapi" in api["name"].lower()),
                    None,
                )
                if wasapi_idx is not None:
                    for i, dev in enumerate(sd.query_devices()):
                        if dev["hostapi"] == wasapi_idx and dev["max_output_channels"] > 0:
                            print(f"  [{i}] {dev['name']} "
                                  f"({dev['max_output_channels']}ch, {int(dev['default_samplerate'])}Hz)")
                else:
                    print("  （找不到 WASAPI host API）")
            except Exception as e:
                print(f"  （無法列出 WASAPI 裝置：{e}）")
        else:
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
    擷取系統播放音訊。

    - Linux:   PipeWire/PulseAudio monitor source（透過 PULSE_SOURCE + ALSA pulse）
    - Windows: WASAPI Loopback（透過 sounddevice WasapiSettings）

    使用 queue.Queue 解耦音訊 callback 與 ASR HTTP 請求，避免
    阻塞操作污染即時音訊執行緒。
    """

    # Linux 預設 monitor source；Windows 為 None（自動偵測預設輸出裝置）
    DEFAULT_DEVICE = None if sys.platform == "win32" else "alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor"
    ALSA_PULSE_DEVICE = "pulse"  # Linux only：ALSA pulse plugin

    def __init__(self, device: str | None = None):
        # Linux: PulseAudio source 名稱（None → DEFAULT_DEVICE）
        # Windows: 輸出裝置名稱或索引（None → 自動偵測預設輸出）
        self._device = device if sys.platform == "win32" else (device or self.DEFAULT_DEVICE)
        self._stream = None
        self._pa = None          # pyaudiowpatch instance (Windows only)
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

        self._callback = callback
        self._buf = np.zeros(0, dtype=np.float32)
        self._running = True

        if sys.platform == "win32":
            self._setup_windows(sd)
        else:
            self._setup_linux(sd)

        # 消費者執行緒：從 queue 取音訊、resample、送 callback
        self._consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self._consumer_thread.start()
        self._stream.start()

    def _setup_linux(self, sd) -> None:
        """Linux：透過 PULSE_SOURCE + ALSA pulse device 擷取 monitor source。"""
        os.environ["PULSE_SOURCE"] = self._device
        dev_info = sd.query_devices(self.ALSA_PULSE_DEVICE, kind="input")
        self._native_sr = int(dev_info["default_samplerate"])  # 通常 44100 或 48000
        self._stream = sd.InputStream(
            samplerate=self._native_sr,
            channels=1,
            dtype="float32",
            blocksize=int(self._native_sr * 0.05),  # 50ms 固定 buffer
            device=self.ALSA_PULSE_DEVICE,
            callback=self._sd_callback,
        )

    def _setup_windows(self, sd) -> None:
        """Windows：透過 pyaudiowpatch WASAPI Loopback 擷取系統播放音訊。"""
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        if self._device is not None:
            loopback_idx = int(self._device)
            dev_info = self._pa.get_device_info_by_index(loopback_idx)
        else:
            # 自動：找預設輸出裝置對應的 loopback 裝置
            default_out_idx = wasapi_info["defaultOutputDevice"]
            default_out = self._pa.get_device_info_by_index(default_out_idx)
            loopback_idx = None
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice") and dev["name"].startswith(default_out["name"]):
                    loopback_idx = i
                    dev_info = dev
                    break
            if loopback_idx is None:
                raise RuntimeError(
                    f"找不到 '{default_out['name']}' 的 WASAPI Loopback 裝置"
                )

        self._native_sr = int(dev_info["defaultSampleRate"])
        channels = max(int(dev_info["maxInputChannels"]), 1)
        print(f"[Monitor] WASAPI Loopback: {dev_info['name']}  sr={self._native_sr}  ch={channels}", flush=True)

        def _pa_callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                audio = audio.reshape(-1, channels)[:, 0]
            self._queue.put(audio.copy())
            return (None, pyaudio.paContinue)

        pa_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=self._native_sr,
            input=True,
            input_device_index=loopback_idx,
            frames_per_buffer=int(self._native_sr * 0.05),
            stream_callback=_pa_callback,
        )

        # 包裝成相容 sounddevice 介面的物件
        class _StreamWrapper:
            def __init__(self, s): self._s = s
            def start(self): self._s.start_stream()
            def stop(self): self._s.stop_stream()
            def close(self): self._s.close()

        self._stream = _StreamWrapper(pa_stream)

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

            try:
                # resample native_sr → 16kHz（在非即時執行緒中進行）
                target_len = int(len(raw) * TARGET_SR / self._native_sr)
                if target_len == 0:
                    continue
                resampled = signal.resample(raw, target_len).astype(np.float32)
                self._buf = np.concatenate([self._buf, resampled])

                # 每累積 CHUNK_SAMPLES 就送出一次
                while len(self._buf) >= CHUNK_SAMPLES:
                    chunk = self._buf[:CHUNK_SAMPLES].copy()
                    self._buf = self._buf[CHUNK_SAMPLES:]
                    if self._callback:
                        self._callback(chunk)
            except Exception as e:
                print(f"[Consumer error] {e}", flush=True)
                import traceback; traceback.print_exc()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        if self._consumer_thread:
            self._consumer_thread.join(timeout=1.0)
            self._consumer_thread = None
        self._buf = np.zeros(0, dtype=np.float32)


class MicrophoneAudioSource(AudioSource):
    """麥克風音訊來源。"""

    def __init__(self, device=None):
        self._device = device  # None = 系統預設麥克風
        self._stream = None
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._native_sr: int = 0
        self._callback: Callable[[np.ndarray], None] | None = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._consumer_thread: threading.Thread | None = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._stream is not None:
            raise RuntimeError("MicrophoneAudioSource is already running; call stop() first.")
        import sounddevice as sd
        dev_info = sd.query_devices(self._device, kind="input")
        self._native_sr = int(dev_info["default_samplerate"])
        self._callback = callback
        self._buf = np.zeros(0, dtype=np.float32)
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self._consumer_thread.start()
        self._stream = sd.InputStream(
            samplerate=self._native_sr,
            channels=1,
            dtype="float32",
            blocksize=int(self._native_sr * 0.05),
            device=self._device,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _sd_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[Audio] {status}")
        self._queue.put(indata[:, 0].copy())

    def _consumer(self) -> None:
        while self._running:
            try:
                raw = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            target_len = int(len(raw) * TARGET_SR / self._native_sr)
            resampled = signal.resample(raw, target_len).astype(np.float32)
            self._buf = np.concatenate([self._buf, resampled])
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

# ---------------------------------------------------------------------------
# Worker Process（音訊 + ASR + 翻譯，無 X11）
# ---------------------------------------------------------------------------

def _worker_main(text_q: multiprocessing.SimpleQueue, cmd_q: multiprocessing.SimpleQueue, cfg: dict) -> None:
    """
    在獨立 subprocess 執行：sounddevice + VAD + ASR + 翻譯。
    完全不使用 X11/tkinter，避免與主程序的 XCB 衝突。

    text_q: 送出 {"original": str, "translated": str} 或 {"direction": str}
    cmd_q:  接收 "toggle"（切換翻譯方向）或 "stop"

    架構：
    - on_chunk：非阻塞，只把音訊放入 _vad_q
    - vad_loop：Silero VAD 偵測語音/靜音，累積語音片段，
                靜音 ~0.8s 後把完整語音放入 _speech_q
    - asr_loop：等待 _speech_q，送到 ASR server，更新字幕
    """
    import onnxruntime as ort
    from pathlib import Path
    import opencc

    os.environ.pop("DISPLAY", None)

    # 簡體→台灣繁體轉換器（s2twp 包含詞彙替換，如「軟件→軟體」）
    _s2tw = opencc.OpenCC("s2twp")

    current_original = ""

    def on_translation(translated: str) -> None:
        text_q.put({"original": current_original, "translated": translated})

    debouncer = TranslationDebouncer(
        api_key=cfg["openai_api_key"],
        callback=on_translation,
        model=cfg["translation_model"],
    )
    debouncer.set_direction(cfg["direction"])

    if cfg["source"] == "monitor":
        audio_source = MonitorAudioSource(device=cfg["monitor_device"])
    else:
        audio_source = MicrophoneAudioSource(device=cfg.get("mic_device"))

    asr = ASRClient(cfg["asr_server"])

    # Silero VAD 常數（v6 模型）
    VAD_CHUNK = 576               # 36ms @ 16kHz
    VAD_THRESHOLD = 0.5
    RT_SILENCE_CHUNKS = 22        # 0.8s - 靜音後觸發轉錄（同 QwenASRMiniTool）
    RT_MAX_BUFFER_CHUNKS = 277    # 10s  - 強制 flush（HTTP timeout 20s 限制，取一半）

    # 載入 VAD 模型
    _vad_model_path = Path(__file__).parent / "silero_vad_v6.onnx"
    vad_sess = ort.InferenceSession(str(_vad_model_path))

    _vad_q: queue.Queue = queue.Queue()
    # _speech_q 傳送 (audio: np.ndarray, event: str)
    # event = "probe" - 短靜音，檢查是否句末再決定要不要顯示
    # event = "force" - 強制 flush（長靜音或 max buffer）
    _speech_q: queue.Queue = queue.Queue()
    _stop_event = threading.Event()

    def on_chunk(audio: np.ndarray) -> None:
        """非阻塞：只把音訊放入 VAD 佇列。"""
        _vad_q.put(audio)

    def vad_loop() -> None:
        """
        VAD 執行緒：靜音偵測。

        語音結束（靜音 ≥ 0.8s）或 buffer 達 10s 上限時，把整段語音送到 _speech_q。
        """
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        buf: list[np.ndarray] = []
        sil_cnt = 0
        leftover = np.zeros(0, dtype=np.float32)

        try:
            while not _stop_event.is_set():
                try:
                    audio = _vad_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                audio = np.concatenate([leftover, audio])
                n_chunks = len(audio) // VAD_CHUNK
                leftover = audio[n_chunks * VAD_CHUNK:]

                for i in range(n_chunks):
                    chunk = audio[i * VAD_CHUNK:(i + 1) * VAD_CHUNK]
                    inp = chunk[np.newaxis, :].astype(np.float32)
                    out = vad_sess.run(
                        ["speech_probs", "hn", "cn"],
                        {"input": inp, "h": h, "c": c},
                    )
                    prob, h, c = out
                    prob = float(prob.flatten()[0])

                    if prob >= VAD_THRESHOLD:
                        buf.append(chunk)
                        sil_cnt = 0
                    elif buf:
                        buf.append(chunk)
                        sil_cnt += 1
                        if sil_cnt >= RT_SILENCE_CHUNKS:
                            # 靜音 0.8s：送出整段語音，保留 h/c 以免下句開頭被漏偵測
                            seg = np.concatenate(buf)
                            print(f"[VAD] flush silence {len(seg)/TARGET_SR:.2f}s", flush=True)
                            _speech_q.put(seg)
                            buf = []
                            sil_cnt = 0

                    # Max buffer 10s：強制送出，保留 h/c
                    if len(buf) >= RT_MAX_BUFFER_CHUNKS:
                        seg = np.concatenate(buf)
                        print(f"[VAD] flush max {len(seg)/TARGET_SR:.2f}s", flush=True)
                        _speech_q.put(seg)
                        buf = []
                        sil_cnt = 0

        except Exception as e:
            print(f"[VAD fatal error] {e}", flush=True)
            import traceback; traceback.print_exc()

    def _to_traditional(text: str, language: str) -> str:
        """若語言為中文（語言標籤或文字內含 CJK），將簡體轉成台灣繁體。"""
        is_chinese = (
            (language and any(kw in language.lower() for kw in ("chinese", "mandarin", "cantonese")))
            or any("\u4e00" <= c <= "\u9fff" for c in text)
        )
        if is_chinese:
            return _s2tw.convert(text)
        return text

    def asr_loop() -> None:
        """ASR 執行緒：one-shot 轉錄，收到整段語音就直接送 server 辨識。"""
        nonlocal current_original
        print("[ASR] thread started", flush=True)

        while not _stop_event.is_set():
            try:
                audio = _speech_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if len(audio) < TARGET_SR // 8:   # < 0.125s，跳過
                continue

            try:
                result = asr.transcribe(audio)
                language = result.get("language", "")
                text = _to_traditional(result.get("text", ""), language)
                print(f"[ASR] lang={language!r} text={text!r} same={text == current_original}", flush=True)

                if text and text != current_original:
                    current_original = text
                    text_q.put({"original": text, "translated": ""})
                    debouncer.update(text)  # 翻譯開啟

            except Exception as e:
                print(f"[Worker ASR error] {e}", flush=True)
                # timeout 後清空積壓的舊 chunk，避免 server 持續過載
                if "timed out" in str(e).lower():
                    drained = 0
                    while not _speech_q.empty():
                        try:
                            _speech_q.get_nowait()
                            drained += 1
                        except queue.Empty:
                            break
                    if drained:
                        print(f"[ASR] Cleared {drained} stale chunks after timeout", flush=True)

    vad_thread = threading.Thread(target=vad_loop, daemon=True, name="vad-thread")
    asr_thread = threading.Thread(target=asr_loop, daemon=True, name="asr-thread")
    vad_thread.start()
    asr_thread.start()

    audio_source.start(on_chunk)
    print("[Worker] Audio capture started.", flush=True)

    try:
        while True:
            if not cmd_q.empty():
                cmd = cmd_q.get()
                if cmd == "toggle":
                    new_dir = debouncer.toggle_direction()
                    text_q.put({"direction": new_dir})
                elif cmd == "switch_source":
                    audio_source.stop()
                    if isinstance(audio_source, MonitorAudioSource):
                        audio_source = MicrophoneAudioSource(device=cfg.get("mic_device"))
                        src_name = "mic"
                    else:
                        audio_source = MonitorAudioSource(device=cfg["monitor_device"])
                        src_name = "monitor"
                    audio_source.start(on_chunk)
                    text_q.put({"source": src_name})
                elif cmd == "stop":
                    break
            else:
                time.sleep(0.1)
    finally:
        _stop_event.set()
        audio_source.stop()
        debouncer.shutdown()
        vad_thread.join(timeout=3)
        asr_thread.join(timeout=5)
        print("[Worker] Stopped.", flush=True)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time subtitle overlay")
    parser.add_argument("--asr-server", default="http://localhost:8000",
                        help="Qwen3-ASR streaming server URL")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--screen", type=int, default=0,
                        help="Display screen index (0=primary, 1=secondary)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--translation-model", default="gpt-4o-mini",
                        help="OpenAI model for translation")
    parser.add_argument("--source", choices=["monitor", "mic"], default="monitor",
                        help="Audio source: monitor（系統音訊）or mic（麥克風）")
    parser.add_argument("--monitor-device", default=MonitorAudioSource.DEFAULT_DEVICE,
                        help="音訊擷取裝置：Linux=PulseAudio monitor source 名稱；"
                             "Windows=WASAPI 輸出裝置名稱或索引（None=自動偵測預設輸出）。"
                             "用 --list-devices 查詢可用裝置")
    parser.add_argument("--mic-device", default=None,
                        help="麥克風裝置名稱或索引（None = 系統預設麥克風）")
    parser.add_argument("--direction", choices=["en→zh", "zh→en"], default="en→zh",
                        help="Initial translation direction")
    args = parser.parse_args()

    if args.list_devices:
        AudioSource.list_devices()
        return

    if not args.openai_api_key:
        print("Error: --openai-api-key 或 OPENAI_API_KEY 環境變數必須設定")
        return

    cfg = {
        "asr_server": args.asr_server,
        "openai_api_key": args.openai_api_key,
        "translation_model": args.translation_model,
        "source": args.source,
        "monitor_device": args.monitor_device,
        "mic_device": args.mic_device,
        "direction": args.direction,
    }

    # 準備 IPC queues（用 SimpleQueue，不會在主程序產生 feeder 背景執行緒）
    text_q: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()
    cmd_q: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()

    # 本地方向追蹤（UI 用，與 worker 同步）
    current_direction = [args.direction]

    def on_toggle() -> str:
        current_direction[0] = "zh→en" if current_direction[0] == "en→zh" else "en→zh"
        cmd_q.put("toggle")
        return current_direction[0]

    def on_switch_source() -> None:
        cmd_q.put("switch_source")

    # 先建立 tkinter（在 fork 之前完成 X11 連線，child 繼承 fd 但立即移除 DISPLAY）
    overlay = SubtitleOverlay(
        screen_index=args.screen,
        on_toggle_direction=on_toggle,
        on_switch_source=on_switch_source,
    )
    overlay.update_direction_label(args.direction)

    # tkinter 初始化後才 fork worker（child 不使用 X11）
    worker = multiprocessing.Process(
        target=_worker_main, args=(text_q, cmd_q, cfg),
        daemon=True, name="subtitle-worker",
    )
    worker.start()

    # 用 tkinter after() 輪詢 text_q（全在主執行緒，零 X11 競爭）
    _last_translated = [""]  # 保留上一筆翻譯，直到新翻譯到來才替換

    def poll() -> None:
        while not text_q.empty():
            msg = text_q.get()
            if "direction" in msg:
                overlay.update_direction_label(msg["direction"])
            elif "source" in msg:
                overlay.update_source_label(msg["source"])
            else:
                translated = msg.get("translated", "")
                if translated:
                    _last_translated[0] = translated
                overlay.set_text(
                    original=msg.get("original", ""),
                    translated=_last_translated[0],
                )
        overlay._root.after(50, poll)

    overlay._root.after(50, poll)
    overlay.run()  # blocking，直到視窗關閉

    # 視窗關閉後停止 worker
    cmd_q.put("stop")
    worker.join(timeout=3)
    if worker.is_alive():
        worker.terminate()


if __name__ == "__main__":
    # spawn：全新 Python 程序，不繼承 X11 socket fd，避免 XCB 序號衝突
    multiprocessing.set_start_method("spawn")
    main()
