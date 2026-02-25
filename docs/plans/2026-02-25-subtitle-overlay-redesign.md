# Subtitle Overlay Redesign — Draggable, Resizable, Transparent

Date: 2026-02-25

## Goal

Replace the fixed full-width bottom overlay with a freely movable, resizable, fully transparent window that shows only the subtitle text (with readable shadow/outline), minimising interference with the user's screen.

## Approach: Canvas + `-transparentcolor`

### Window Setup

- Keep `overrideredirect(True)` and `-topmost True`
- Set `bg="black"` on root and Canvas
- Apply `wm_attributes("-transparentcolor", "black")` — Windows renders all black pixels as fully transparent, so the window has no visible background; only non-black pixels (text, handles) are visible
- Default size: 900 × 120 px, default position: bottom-centre of primary screen

### Text Rendering (Canvas)

Use a `tk.Canvas` filling the entire window instead of `tk.Label` widgets.

For each text line, draw the shadow first then the foreground:

```
# shadow  (offset +2, +2, colour #000000)
canvas.create_text(x+2, y+2, text=text, fill="#000000", font=font, anchor="w")
# foreground
canvas.create_text(x,   y,   text=text, fill="#ffffff", font=font, anchor="w")
```

- Line 1 (original/EN): font Arial 15, fg `#dddddd`, shadow `#000000`
- Line 2 (translated/ZH): font Microsoft JhengHei 22 bold, fg `#ffffff`, shadow `#000000`

### Drag to Move

Bind to the Canvas:

- `<ButtonPress-1>` on canvas body → record `(event.x_root - win.winfo_x(), event.y_root - win.winfo_y())` as `_drag_offset`
- `<B1-Motion>` → `geometry(f"+{event.x_root - dx}+{event.y_root - dy}")`

Right-corner resize handle intercepts events first (see below), so drag must only trigger when NOT on the resize zone.

### Resize Handle

A 16 × 16 px semi-white triangle drawn at the bottom-right corner of the Canvas. Separate bindings:

- `<ButtonPress-1>` on handle → record current size + mouse position
- `<B1-Motion>` → compute new width/height, apply `geometry(f"{w}x{h}+{x}+{y}")`; minimum 300 × 80

After resize, re-draw all text items (or just re-anchor them to new canvas size).

### Hover Toolbar

A `tk.Frame` (height 28, `bg="#222222"`, `alpha` unchanged since it's non-black) overlaid at the top of the window. Contains the same buttons as today (direction toggle, source toggle, close).

- Normally hidden: `toolbar.place_forget()`
- Show on `<Enter>` to root window; hide on `<Leave>` with a 400 ms delay (cancel if re-entered)

The toolbar background is `#222222` (dark grey, not pure black) so it is **not** made transparent by `-transparentcolor`.

### Colour Summary

| Element | Colour |
|---------|--------|
| Window / Canvas bg | `#000000` → fully transparent |
| EN text | `#dddddd` |
| ZH text | `#ffffff` |
| Text shadow | `#000000` (transparent — acceptable; only text outline matters) |
| Resize handle | `#888888` (semi-visible triangle) |
| Toolbar bg | `#222222` |
| Toolbar buttons | bg `#333333`, fg `#ffffff` |

> **Note on shadow colour**: Because black is transparent, the shadow layer is also invisible. To compensate, draw the shadow in `#111111` (almost black but not fully) so it renders as a very dark nearly-opaque pixel on Windows.

### Architecture Changes (SubtitleOverlay class)

1. Remove Label widgets for EN/ZH text → replace with Canvas + `create_text` calls
2. Add `_start_drag`, `_do_drag` methods for window movement
3. Add `_start_resize`, `_do_resize` methods + `_draw_resize_handle()`
4. Add `_show_toolbar`, `_hide_toolbar` methods with delayed hide
5. `set_text()` → call `_redraw_text()` which clears and re-creates canvas text items
6. Add `_redraw_text()` that positions text based on current canvas size

### No New Dependencies

Everything uses the standard `tkinter` library already in use.

## Out of Scope

- Saving window position/size across sessions (future work)
- Per-monitor DPI scaling
- Linux implementation changes (Linux path unchanged)
