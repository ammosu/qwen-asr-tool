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
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def finish(self) -> dict:
        """結束 session，回傳最終結果。"""
        assert self.session_id, "Call start() first"
        sid = self.session_id
        self.session_id = None
        r = requests.post(
            f"{self.base_url}/api/finish",
            params={"session_id": sid},
            timeout=60,
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

    def __init__(self, screen_index: int = 0, on_toggle_direction=None, on_switch_source=None):
        self._on_toggle_direction = on_toggle_direction
        self._on_switch_source = on_switch_source

        self._root = tk.Tk()

        # 用 tkinter 取螢幕尺寸（不依賴 screeninfo）
        self._width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        self._x = 0
        self._y = screen_height - self.WINDOW_HEIGHT

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

        self._root.bind("<Escape>", lambda e: self._do_close())
        self._root.bind("<F9>", lambda e: self._toggle_direction())
        self._root.protocol("WM_DELETE_WINDOW", self._do_close)

    def _do_close(self):
        """關閉視窗。"""
        self._root.destroy()

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
            self._en_var.set(original[-120:] if len(original) > 120 else original)
            self._zh_var.set(translated[-60:] if len(translated) > 60 else translated)
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
    RT_SILENCE_CHUNKS = 22        # 0.8s - 短靜音：probe 句末
    RT_LONG_SILENCE_CHUNKS = 55   # 2s   - 長靜音：強制 flush
    RT_MAX_BUFFER_CHUNKS = 83     # 3s   - 強制 flush（限制單次 push_chunk ≤ 3s，避免 server timeout）

    # 句末符號（ASR 回傳英文標點或中文標點皆可）
    SENTENCE_END_CHARS = frozenset('.?!。？！…')

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
        VAD 執行緒：兩段式靜音偵測。

        短靜音（0.8s）→ 送 (buf, "probe")，由 asr_loop 決定是否句末
        長靜音（2s）  → 送 (empty, "force")，強制顯示
        max buffer   → 送 (buf, "force")，強制顯示
        """
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        buf: list[np.ndarray] = []
        sil_cnt = 0
        probed = False   # 是否已送出 probe（等待 long silence 或新語音）
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
                        probed = False
                    elif buf or sil_cnt > 0:
                        if buf:
                            buf.append(chunk)
                        sil_cnt += 1

                        if not probed and sil_cnt >= RT_SILENCE_CHUNKS:
                            # 短靜音：送 probe，清空 buf 但保留 session
                            probe_audio = np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
                            _speech_q.put((probe_audio, "probe"))
                            buf = []
                            probed = True
                        elif probed and sil_cnt >= RT_LONG_SILENCE_CHUNKS:
                            # 長靜音：強制 flush
                            _speech_q.put((np.zeros(0, dtype=np.float32), "force"))
                            sil_cnt = 0
                            probed = False
                            h = np.zeros((1, 1, 128), dtype=np.float32)
                            c = np.zeros((1, 1, 128), dtype=np.float32)

                    # Max buffer：強制 flush
                    if len(buf) >= RT_MAX_BUFFER_CHUNKS:
                        _speech_q.put((np.concatenate(buf), "force"))
                        buf = []
                        sil_cnt = 0
                        probed = False
                        h = np.zeros((1, 1, 128), dtype=np.float32)
                        c = np.zeros((1, 1, 128), dtype=np.float32)

        except Exception as e:
            print(f"[VAD fatal error] {e}", flush=True)
            import traceback; traceback.print_exc()

    def _parse_asr_result(raw: str) -> tuple[str, str]:
        """剝除 'language XXX<asr_text>' 前綴，回傳 (language, text)。"""
        language = ""
        if raw.startswith("language ") and "<asr_text>" in raw:
            header, text = raw.split("<asr_text>", 1)
            language = header.removeprefix("language ").strip()
            return language, text.strip()
        if "<asr_text>" in raw:
            return "", raw.split("<asr_text>", 1)[1].strip()
        return "", raw.strip()

    def _to_traditional(text: str, language: str) -> str:
        """若語言為中文，將簡體轉成台灣繁體。"""
        if language and "chinese" in language.lower():
            return _s2tw.convert(text)
        return text

    def asr_loop() -> None:
        """
        ASR 執行緒：句子組合器（sentence assembler）。

        每個 2s 片段辨識後累積到 assembled_parts。
        只有當組合後文字以句末符號結尾，或已累積 ≥ MAX_ASSEMBLE_PARTS 個片段時，
        才顯示組合後的完整句子。

        probe  + 句末符號  → 立即顯示組合結果
        force              → 累積到組合器，句末才顯示（或片段數達上限）
        """
        nonlocal current_original
        assembled_parts: list[str] = []   # 等待組合的片段
        MAX_ASSEMBLE_PARTS = 3            # 最多累積 ~6s 的片段再強制顯示

        while not _stop_event.is_set():
            try:
                audio, event = _speech_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if len(audio) < TARGET_SR // 8:   # < 0.125s，跳過
                continue

            try:
                asr.start()
                result = asr.push_chunk(audio)
                inter_lang, intermediate_text = _parse_asr_result(result.get("text", ""))
                try:
                    fin = asr.finish()
                    fin_lang, fin_text = _parse_asr_result(fin.get("text", ""))
                    language = fin_lang or inter_lang
                    text = fin_text or intermediate_text
                except Exception:
                    language = inter_lang
                    text = intermediate_text

                text = _to_traditional(text, language)

                if not text:
                    continue

                assembled_parts.append(text)
                assembled = " ".join(assembled_parts)

                # 判斷是否顯示：句末符號 OR 達到片段上限
                sentence_done = assembled[-1] in SENTENCE_END_CHARS
                force_show = len(assembled_parts) >= MAX_ASSEMBLE_PARTS

                if sentence_done or force_show or event == "probe":
                    if assembled != current_original:
                        current_original = assembled
                        text_q.put({"original": assembled, "translated": ""})
                        # debouncer.update(assembled)  # 翻譯暫時關閉
                    assembled_parts = []

            except Exception as e:
                print(f"[Worker ASR error] {e}", flush=True)
                try:
                    asr.finish()
                except Exception:
                    pass

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
        try:
            asr.finish()
        except Exception:
            pass
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
                        help="PulseAudio monitor source name（用 --list-devices 查詢）")
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
    def poll() -> None:
        while not text_q.empty():
            msg = text_q.get()
            if "direction" in msg:
                overlay.update_direction_label(msg["direction"])
            elif "source" in msg:
                overlay.update_source_label(msg["source"])
            else:
                overlay.set_text(
                    original=msg.get("original", ""),
                    translated=msg.get("translated", ""),
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
