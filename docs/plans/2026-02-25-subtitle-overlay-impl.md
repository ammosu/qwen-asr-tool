# Subtitle Overlay Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed full-width black overlay with a fully transparent, draggable, resizable subtitle window that shows only shadow-outlined text.

**Architecture:** Use tkinter's `-transparentcolor black` to make the black canvas background transparent; render text via `canvas.create_text()` with a near-black shadow layer for readability; implement drag/resize via `<ButtonPress-1>` / `<B1-Motion>` bindings; show toolbar only on mouse hover.

**Tech Stack:** Python 3.12, tkinter (stdlib), no new dependencies.

**File:** `subtitle_client/subtitle_client.py` — only `SubtitleOverlay` class (lines ~177–316) is touched.

---

### Task 1: Replace Label widgets with Canvas

**Files:**
- Modify: `subtitle_client/subtitle_client.py` (SubtitleOverlay.__init__)

**Step 1: Remove Label widgets and add Canvas**

In `__init__`, delete the two `tk.Label(...)` blocks for `_en_var` and `_zh_var`.
Replace with a single Canvas that fills the window:

```python
self._canvas = tk.Canvas(
    self._root,
    bg=self.BG_COLOR,        # "#000000" → transparent via transparentcolor
    highlightthickness=0,
)
self._canvas.pack(fill="both", expand=True)
```

Keep `_en_var` and `_zh_var` StringVars for now (they'll be removed in Task 2).

**Step 2: Add transparentcolor attribute**

In `__init__`, after `wm_attributes("-alpha", 0.85)`, add:

```python
self._root.wm_attributes("-transparentcolor", self.BG_COLOR)
```

And remove the `-alpha` line (transparency now comes from the transparentcolor trick, not window-level alpha):

```python
# DELETE this line:
self._root.wm_attributes("-alpha", 0.85)
# ADD this line:
self._root.wm_attributes("-transparentcolor", self.BG_COLOR)
```

**Step 3: Update colour constants**

Change class-level constants:

```python
TOOLBAR_BG  = "#222222"   # dark grey — NOT black, so toolbar remains visible
BG_COLOR    = "#000000"   # fully transparent via transparentcolor
EN_COLOR    = "#dddddd"
ZH_COLOR    = "#ffffff"
SHADOW_COLOR = "#111111"  # near-black shadow (not #000000 which would be transparent)
```

Remove `EN_FONT` and `ZH_FONT` class constants (they move into `_redraw_text`).

**Step 4: Verify window launches**

```bash
cd subtitle_client
python subtitle_client.py --asr-server http://10.0.0.85:8000 --source monitor --openai-api-key sk-dummy
```

Expected: window appears, background is transparent (you can see desktop through it), toolbar visible at top.

---

### Task 2: Implement `_redraw_text()` on Canvas

**Files:**
- Modify: `subtitle_client/subtitle_client.py`

**Step 1: Add `_redraw_text` method to SubtitleOverlay**

```python
def _redraw_text(self):
    """Clear canvas and re-draw subtitle text with shadow."""
    self._canvas.delete("text")   # tag="text" for all text items

    w = self._canvas.winfo_width() or self._root.winfo_width()

    en_text = self._en_str
    zh_text = self._zh_str

    # EN line — 20px from left, 12px from top of canvas area
    ex, ey = 20, 12
    en_font = ("Arial", 15)
    zh_font = ("Microsoft JhengHei", 22, "bold")

    # Shadow layers (near-black, offset +2/+2)
    self._canvas.create_text(ex+2, ey+2, text=en_text, fill=self.SHADOW_COLOR,
                             font=en_font, anchor="nw", width=w-40, tags="text")
    self._canvas.create_text(ex, ey, text=en_text, fill=self.EN_COLOR,
                             font=en_font, anchor="nw", width=w-40, tags="text")

    # ZH line — below EN, estimate EN height ~22px
    zy = ey + 30
    self._canvas.create_text(ex+2, zy+2, text=zh_text, fill=self.SHADOW_COLOR,
                             font=zh_font, anchor="nw", width=w-40, tags="text")
    self._canvas.create_text(ex, zy, text=zh_text, fill=self.ZH_COLOR,
                             font=zh_font, anchor="nw", width=w-40, tags="text")
```

**Step 2: Add instance variables in `__init__`**

Before creating Canvas, add:

```python
self._en_str = ""
self._zh_str = ""
```

**Step 3: Update `set_text()` to use `_redraw_text`**

Replace the existing `set_text` method:

```python
def set_text(self, original: str = "", translated: str = ""):
    def _update():
        self._en_str = original[-120:] if len(original) > 120 else original
        self._zh_str = translated[-60:] if len(translated) > 60 else translated
        self._redraw_text()
    self._root.after(0, _update)
```

Remove `_en_var` and `_zh_var` StringVars entirely.

**Step 4: Bind `<Configure>` for resize redraws**

In `__init__`, after canvas creation:

```python
self._canvas.bind("<Configure>", lambda e: self._redraw_text())
```

**Step 5: Verify text renders**

Launch client, play audio. Text should appear as white/grey with subtle dark shadow, fully readable against any background.

---

### Task 3: Add drag-to-move

**Files:**
- Modify: `subtitle_client/subtitle_client.py`

**Step 1: Add drag state variables in `__init__`**

```python
self._drag_x = 0
self._drag_y = 0
```

**Step 2: Add drag methods**

```python
def _start_drag(self, event):
    self._drag_x = event.x_root - self._root.winfo_x()
    self._drag_y = event.y_root - self._root.winfo_y()

def _do_drag(self, event):
    nx = event.x_root - self._drag_x
    ny = event.y_root - self._drag_y
    self._root.geometry(f"+{nx}+{ny}")
```

**Step 3: Bind drag to canvas in `__init__`**

```python
self._canvas.bind("<ButtonPress-1>", self._start_drag)
self._canvas.bind("<B1-Motion>", self._do_drag)
```

**Step 4: Verify drag works**

Launch client, click and drag anywhere on the subtitle text area — window should follow the mouse freely.

---

### Task 4: Add resize handle

**Files:**
- Modify: `subtitle_client/subtitle_client.py`

**Step 1: Add resize state variables in `__init__`**

```python
self._resize_start = None   # (mouse_x, mouse_y, win_w, win_h)
```

**Step 2: Add `_draw_resize_handle` method**

```python
RESIZE_SIZE = 16

def _draw_resize_handle(self):
    """Draw a small triangle at bottom-right of canvas."""
    self._canvas.delete("resize_handle")
    w = self._canvas.winfo_width() or self._root.winfo_width()
    h = self._canvas.winfo_height() or self._root.winfo_height()
    s = self.RESIZE_SIZE
    # Draw a small triangle in dark grey (non-black so it's visible)
    self._canvas.create_polygon(
        w, h-s, w-s, h, w, h,
        fill="#888888", outline="", tags="resize_handle"
    )
```

Call `_draw_resize_handle()` inside `_redraw_text()` at the end.

**Step 3: Add resize methods**

```python
def _start_resize(self, event):
    self._resize_start = (
        event.x_root, event.y_root,
        self._root.winfo_width(), self._root.winfo_height(),
    )
    # Prevent drag from triggering
    return "break"

def _do_resize(self, event):
    if not self._resize_start:
        return
    mx0, my0, w0, h0 = self._resize_start
    dw = event.x_root - mx0
    dh = event.y_root - my0
    new_w = max(300, w0 + dw)
    new_h = max(80, h0 + dh)
    x = self._root.winfo_x()
    y = self._root.winfo_y()
    self._root.geometry(f"{new_w}x{new_h}+{x}+{y}")
    return "break"

def _stop_resize(self, event):
    self._resize_start = None
```

**Step 4: Bind resize to handle tag in `__init__`**

```python
self._canvas.tag_bind("resize_handle", "<ButtonPress-1>",  self._start_resize)
self._canvas.tag_bind("resize_handle", "<B1-Motion>",      self._do_resize)
self._canvas.tag_bind("resize_handle", "<ButtonRelease-1>", self._stop_resize)
```

**Step 5: Verify resize**

Launch client, drag the grey triangle at bottom-right — window should grow/shrink. Minimum size 300×80 enforced.

---

### Task 5: Hover toolbar (show on mouse enter, hide on leave)

**Files:**
- Modify: `subtitle_client/subtitle_client.py`

**Step 1: Convert toolbar to overlay using `place()`**

In `__init__`, change toolbar creation to use `place` instead of `pack`:

```python
toolbar = tk.Frame(self._root, bg=self.TOOLBAR_BG, height=self.TOOLBAR_HEIGHT)
# Do NOT pack it — place it as an overlay
toolbar.place(x=0, y=0, relwidth=1.0, height=self.TOOLBAR_HEIGHT)
toolbar.place_forget()   # hidden by default
self._toolbar = toolbar
```

Move canvas creation BEFORE toolbar so canvas is lower in z-order:

```python
# Order in __init__:
# 1. create canvas (pack fill=both, expand=True)
# 2. create toolbar frame (place, then place_forget)
# 3. add buttons to toolbar
```

**Step 2: Add show/hide methods with delay**

```python
self._toolbar_hide_id = None

def _show_toolbar(self, event=None):
    if self._toolbar_hide_id:
        self._root.after_cancel(self._toolbar_hide_id)
        self._toolbar_hide_id = None
    self._toolbar.place(x=0, y=0, relwidth=1.0, height=self.TOOLBAR_HEIGHT)

def _hide_toolbar(self, event=None):
    self._toolbar_hide_id = self._root.after(
        400, lambda: self._toolbar.place_forget()
    )
```

**Step 3: Bind hover events in `__init__`**

```python
self._root.bind("<Enter>", self._show_toolbar)
self._root.bind("<Leave>", self._hide_toolbar)
self._toolbar.bind("<Enter>", self._show_toolbar)
self._toolbar.bind("<Leave>", self._hide_toolbar)
```

**Step 4: Verify hover behaviour**

Launch client — toolbar should be invisible. Move mouse over the window → toolbar appears. Move away → hides after ~400 ms.

---

### Task 6: Default window size and position

**Files:**
- Modify: `subtitle_client/subtitle_client.py`

**Step 1: Update geometry constants and `__init__`**

```python
WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 120   # taller canvas now; toolbar overlays on top
```

Update position calculation:

```python
screen_w = self._root.winfo_screenwidth()
screen_h = self._root.winfo_screenheight()
self._x = (screen_w - self.WINDOW_WIDTH) // 2   # horizontally centred
self._y = screen_h - self.WINDOW_HEIGHT - 40     # near bottom, not flush

self._root.geometry(
    f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+{self._x}+{self._y}"
)
```

Remove the old `self._width` usage (no longer full-width).

**Step 2: Verify final layout**

Launch client, confirm:
- Window centred near bottom, not full-width
- Background fully transparent
- Text visible with shadow
- Drag works
- Resize works
- Toolbar appears on hover

---

### Task 7: Commit everything

```bash
cd /c/Users/pang2/qwen-asr-tool
git add subtitle_client/subtitle_client.py docs/plans/
git commit -m "feat(subtitle_client): draggable resizable transparent overlay"
git push
```
