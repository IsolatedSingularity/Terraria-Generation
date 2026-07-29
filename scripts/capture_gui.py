"""Capture the real Tk desktop window for README documentation."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
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
    root.geometry("1500x820+10+10")
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    output = Path(__file__).resolve().parents[1] / "docs" / "media" / "gui.png"

    def capture() -> None:
        app._position_panes()
        app.footer.tkraise()
        root.update_idletasks()
        if sys.platform == "win32":
            rect = ctypes.wintypes.RECT()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)  # type: ignore[attr-defined]
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))  # type: ignore[attr-defined]
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        else:
            left = root.winfo_rootx()
            top = root.winfo_rooty()
            right = left + root.winfo_width()
            bottom = top + root.winfo_height()
        ImageGrab.grab((left, top, right, bottom), all_screens=True).save(output, optimize=True)
        root.destroy()

    def frame_world() -> None:
        if app.current_world is not None:
            app._fit()
            app._center_view()

    root.after(3200, frame_world)
    root.after(4300, capture)
    root.mainloop()
    print(f"Captured {output}")


if __name__ == "__main__":
    main()
