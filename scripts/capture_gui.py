"""Capture the real Tk desktop window for README documentation."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from terraforge.gui import TerraForgeApp


def main() -> None:
    root = tk.Tk()
    app = TerraForgeApp(root)
    root.geometry("1360x820+20+20")
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    output = Path(__file__).resolve().parents[1] / "docs" / "media" / "gui.png"

    def capture() -> None:
        root.update_idletasks()
        left = root.winfo_rootx()
        top = root.winfo_rooty()
        right = left + root.winfo_width()
        bottom = top + root.winfo_height()
        ImageGrab.grab((left, top, right, bottom), all_screens=True).save(output, optimize=True)
        root.destroy()

    def frame_world() -> None:
        if app.current_world is not None:
            app.display_scale = 4
            app._render_view()
            app.canvas.xview_moveto(0.0)
            app.canvas.yview_moveto(0.0)

    root.after(1900, frame_world)
    root.after(2600, capture)
    root.mainloop()
    print(f"Captured {output}")


if __name__ == "__main__":
    main()
