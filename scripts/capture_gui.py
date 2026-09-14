"""Capture the real Tk desktop window for README documentation."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import time
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from terraexplorer.gui import TerraExplorerApp


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    root = tk.Tk()
    app = TerraExplorerApp(root)
    width = min(1500, root.winfo_screenwidth() - 40)
    height = min(820, root.winfo_screenheight() - 90)
    root.geometry(f"{width}x{height}+10+10")
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    output = Path(__file__).resolve().parents[1] / "docs" / "media" / "gui.png"
    started = time.monotonic()
    captured = False

    def capture() -> None:
        nonlocal captured
        app._position_panes()
        app.footer.tkraise()
        root.update_idletasks()
        if sys.platform == "win32":
            rect = ctypes.wintypes.RECT()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)  # type: ignore[attr-defined]
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))  # type: ignore[attr-defined]
            # Visible frame bounds exclude the transparent resize border,
            # which otherwise captures strips of unrelated desktop content.
            ctypes.windll.dwmapi.DwmGetWindowAttribute(  # type: ignore[attr-defined]
                hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)
            )
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        else:
            left = root.winfo_rootx()
            top = root.winfo_rooty()
            right = left + root.winfo_width()
            bottom = top + root.winfo_height()
        if not (
            0 <= left < right <= root.winfo_screenwidth()
            and 0 <= top < bottom <= root.winfo_screenheight()
        ):
            root.destroy()
            raise RuntimeError("Window would be clipped by the desktop; screenshot not saved")
        ImageGrab.grab((left, top, right, bottom), all_screens=True).save(output, optimize=True)
        captured = True
        root.destroy()

    def frame_world() -> None:
        if app.current_world is not None and app.worker is not None and not app.worker.is_alive():
            app._fit()
            app._center_view()
            root.after(1000, capture)
        elif time.monotonic() - started < 180:
            root.after(250, frame_world)
        else:
            root.destroy()
            raise RuntimeError("Generation did not complete; screenshot not saved")

    root.after(1000, frame_world)
    root.mainloop()
    if not captured:
        raise RuntimeError("No completed GUI capture was produced")
    print(f"Captured {output}")


if __name__ == "__main__":
    main()
