"""Capture the real Tk desktop window for README documentation."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from terraforge.gui import TerraForgeApp


def main() -> None:
    root = tk.Tk()
    TerraForgeApp(root)
    root.geometry("1480x900+20+20")
    output = Path(__file__).resolve().parents[1] / "docs" / "media" / "gui.png"

    def capture() -> None:
        root.update_idletasks()
        left = root.winfo_rootx()
        top = root.winfo_rooty()
        right = left + root.winfo_width()
        bottom = top + root.winfo_height()
        ImageGrab.grab((left, top, right, bottom), all_screens=True).save(output, optimize=True)
        root.destroy()

    root.after(2200, capture)
    root.mainloop()
    print(f"Captured {output}")


if __name__ == "__main__":
    main()
