"""Lightweight native Tkinter desktop interface for TerraForge."""

from __future__ import annotations

import queue
import secrets
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from terraforge.config import Difficulty, Evil, WorldConfig, WorldScale
from terraforge.model import GeneratedWorld
from terraforge.passes import Phase
from terraforge.pipeline import GenerationCancelledError, PassEvent, TerraForgePipeline
from terraforge.render import render_world, save_generation_gif, save_npz, save_png
from terraforge.tiles import BIOME_NAMES, TILE_STYLES, Biome, Liquid, Tile, Wall

_DARK_THEME = {
    "bg": "#100d0a",
    "panel": "#1c1712",
    "panel_2": "#2a2118",
    "text": "#f1e5c8",
    "muted": "#b5a27f",
    "accent": "#d09a45",
    "danger": "#bd4d3d",
    "canvas": "#07101a",
}
_LIGHT_THEME = {
    "bg": "#d8cbb1",
    "panel": "#f0e5cf",
    "panel_2": "#d2bea0",
    "text": "#2c1f15",
    "muted": "#6f5b3e",
    "accent": "#8a5a20",
    "danger": "#9a3e32",
    "canvas": "#b6d2dc",
}


class TerraForgeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.light_mode_var = tk.BooleanVar(value=False)
        self.colors = _DARK_THEME
        self.root.title("TerraForge | Clockwork World Forge")
        self.root.geometry("1440x860")
        self.root.minsize(1120, 700)
        self.root.configure(bg=self.colors["bg"])
        self._configure_styles()

        self.current_world: GeneratedWorld | None = None
        self.previous_world: GeneratedWorld | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 4
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.seed_var = tk.StringVar(value="TerraForge")
        self.scale_var = tk.StringVar(value=WorldScale.PREVIEW.value)
        self.evil_var = tk.StringVar(value=Evil.CORRUPTION.value)
        self.difficulty_var = tk.StringVar(value=Difficulty.CLASSIC.value)
        self.hardmode_var = tk.BooleanVar(value=False)
        self.layers_var = tk.BooleanVar(value=True)
        self.biomes_var = tk.BooleanVar(value=False)
        self.markers_var = tk.BooleanVar(value=True)
        self.compare_var = tk.StringVar(value="Current")
        self.status_var = tk.StringVar(
            value="The forge is cold. Choose a seed and wake the machine."
        )
        self.inspector_var = tk.StringVar(value="Click a tile to inspect it")
        self.phase_vars = {phase.value: tk.BooleanVar(value=True) for phase in Phase}

        self._build_layout()
        self.root.after(60, self._poll_events)
        self.root.after(120, self.generate)

    def _configure_styles(self) -> None:
        colors = self.colors
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".",
            background=colors["panel"],
            foreground=colors["text"],
            fieldbackground=colors["panel_2"],
        )
        style.configure("TFrame", background=colors["panel"])
        style.configure(
            "Section.TLabel",
            background=colors["panel"],
            foreground=colors["accent"],
            font=("Georgia", 10, "bold"),
        )
        style.configure("Card.TFrame", background=colors["panel_2"])
        style.configure("TLabel", background=colors["panel"], foreground=colors["text"])
        style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"])
        style.configure(
            "Title.TLabel",
            background=colors["bg"],
            foreground=colors["text"],
            font=("Georgia", 21, "bold"),
        )
        style.configure(
            "Brand.TLabel",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Georgia", 11, "italic"),
        )
        style.configure(
            "TButton",
            background=colors["panel_2"],
            foreground=colors["text"],
            borderwidth=1,
            padding=(10, 7),
            font=("Georgia", 9, "bold"),
        )
        style.map("TButton", background=[("active", colors["accent"])])
        style.configure(
            "Accent.TButton",
            background=colors["accent"],
            foreground=colors["bg"],
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", colors["accent"])])
        style.configure("Danger.TButton", background="#5c2933", foreground="#ffdce2")
        style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"])
        style.configure(
            "TCombobox",
            fieldbackground=colors["panel_2"],
            foreground=colors["text"],
            arrowcolor=colors["text"],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=colors["accent"],
            troughcolor=colors["panel_2"],
        )
        style.configure(
            "Treeview",
            background=colors["panel_2"],
            fieldbackground=colors["panel_2"],
            foreground=colors["text"],
            rowheight=25,
        )
        style.configure("Treeview.Heading", background=colors["panel_2"], foreground=colors["text"])
        style.map("Treeview", background=[("selected", colors["accent"])])

    def _build_layout(self) -> None:
        self.header = tk.Frame(
            self.root,
            bg=self.colors["bg"],
            height=72,
            highlightthickness=1,
            highlightbackground=self.colors["accent"],
        )
        self.header.pack(fill="x", padx=18, pady=(12, 8))
        self._build_brand(self.header)
        ttk.Label(self.header, text="TerraForge", style="Title.TLabel").pack(
            side="left", padx=(12, 6)
        )
        ttk.Label(
            self.header,
            text="seeded worlds, forged one pass at a time",
            style="Brand.TLabel",
        ).pack(side="left", pady=(8, 0))
        ttk.Label(
            self.header,
            text="DEPTH 0 | where every adventure begins",
            style="Muted.TLabel",
        ).pack(side="right", padx=8)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        controls_shell = ttk.Frame(body, width=260)
        viewer = ttk.Frame(body)
        details = ttk.Frame(body, width=350)
        body.add(controls_shell, weight=0)
        body.add(viewer, weight=1)
        body.add(details, weight=0)
        controls = self._scrollable_controls(controls_shell)
        self._build_controls(controls)
        self._build_viewer(viewer)
        self._build_details(details)

        self.footer = tk.Frame(self.root, bg=self.colors["bg"])
        self.footer.pack(fill="x", padx=18, pady=(0, 12))
        self.progress = ttk.Progressbar(self.footer, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ttk.Label(self.footer, textvariable=self.status_var, style="Muted.TLabel").pack(
            side="right"
        )

    def _scrollable_controls(self, parent: ttk.Frame) -> ttk.Frame:
        self.controls_canvas = tk.Canvas(
            parent,
            width=244,
            bg=self.colors["panel"],
            highlightthickness=1,
            highlightbackground=self.colors["accent"],
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.controls_canvas.yview)
        controls = ttk.Frame(self.controls_canvas)
        window = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        controls.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(window, width=event.width),
        )
        self.controls_frame = controls
        self.root.bind_all("<MouseWheel>", self._sidebar_mousewheel, add="+")
        return controls

    def _sidebar_mousewheel(self, event: tk.Event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self.controls_frame:
                direction = -1 if event.delta > 0 else 1
                self.controls_canvas.yview_scroll(direction * 3, "units")
                return
            widget = widget.master

    def _build_brand(self, parent: tk.Widget) -> None:
        logo_path = Path(__file__).parent / "assets" / "terraforge_logo.png"
        logo = Image.open(logo_path).convert("RGBA").resize((56, 56), Image.Resampling.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo)
        self.logo_label = tk.Label(
            parent,
            image=self.logo_photo,
            bg=self.colors["bg"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.logo_label.pack(side="left")

    def _section(self, parent: tk.Widget, title: str) -> ttk.Frame:
        ttk.Label(parent, text=title, style="Section.TLabel").pack(
            anchor="w", padx=10, pady=(13, 5)
        )
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10)
        return frame

    def _build_controls(self, parent: ttk.Frame) -> None:
        world = self._section(parent, "WORLD FORGE")
        ttk.Label(world, text="Seed", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(world, textvariable=self.seed_var).pack(fill="x", pady=(2, 7))
        ttk.Button(world, text="Roll a new seed", command=self.randomize_seed).pack(fill="x")

        for label, variable, values in (
            ("Size", self.scale_var, [item.value for item in WorldScale]),
            ("Evil", self.evil_var, [item.value for item in Evil]),
            ("Difficulty", self.difficulty_var, [item.value for item in Difficulty]),
        ):
            ttk.Label(world, text=label, style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
            ttk.Combobox(world, textvariable=variable, values=values, state="readonly").pack(
                fill="x", pady=(2, 0)
            )
        ttk.Checkbutton(world, text="Apply Hardmode V", variable=self.hardmode_var).pack(
            anchor="w", pady=(8, 0)
        )

        phases = self._section(parent, "WORLD LAYERS")
        for phase in Phase:
            state = "disabled" if phase is Phase.TERRAIN else "normal"
            ttk.Checkbutton(
                phases,
                text=phase.value,
                variable=self.phase_vars[phase.value],
                state=state,
            ).pack(anchor="w")

        view = self._section(parent, "LOOKOUT")
        ttk.Checkbutton(
            view, text="Layer guides", variable=self.layers_var, command=self._render_view
        ).pack(anchor="w")
        ttk.Checkbutton(
            view, text="Biome tint", variable=self.biomes_var, command=self._render_view
        ).pack(anchor="w")
        ttk.Checkbutton(
            view, text="Map symbols", variable=self.markers_var, command=self._render_view
        ).pack(anchor="w")
        ttk.Checkbutton(
            view,
            text="Light interface",
            variable=self.light_mode_var,
            command=self._toggle_theme,
        ).pack(anchor="w")
        ttk.Combobox(
            view,
            textvariable=self.compare_var,
            values=("Current", "Previous", "Split comparison"),
            state="readonly",
        ).pack(fill="x", pady=(6, 0))
        self.compare_var.trace_add("write", lambda *_: self._render_view())

        actions = self._section(parent, "FORGE CONTROLS")
        self.generate_button = ttk.Button(
            actions, text="Forge world", style="Accent.TButton", command=self.generate
        )
        self.generate_button.pack(fill="x", pady=(0, 5))
        self.cancel_button = ttk.Button(
            actions,
            text="Quench forge",
            style="Danger.TButton",
            command=self.cancel,
            state="disabled",
        )
        self.cancel_button.pack(fill="x", pady=(0, 5))
        ttk.Button(actions, text="Export PNG", command=self.export_png).pack(fill="x", pady=2)
        ttk.Button(actions, text="Export generation GIF", command=self.export_gif).pack(
            fill="x", pady=2
        )
        ttk.Button(actions, text="Export NumPy data", command=self.export_npz).pack(
            fill="x", pady=2
        )

    def _build_viewer(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", padx=7, pady=(7, 4))
        ttk.Label(toolbar, text="WORLD WINDOW", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="-", width=3, command=lambda: self._zoom(-1)).pack(
            side="right", padx=2
        )
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom(1)).pack(
            side="right", padx=2
        )
        ttk.Button(toolbar, text="Fit", command=self._fit).pack(side="right", padx=2)

        canvas_frame = ttk.Frame(parent, style="Card.TFrame")
        canvas_frame.pack(fill="both", expand=True, padx=7, pady=(0, 7))
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors["canvas"],
            highlightthickness=2,
            highlightbackground=self.colors["accent"],
        )
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._inspect_tile)
        self.canvas.bind("<MouseWheel>", self._mouse_zoom)

    def _build_details(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="GENERATION LOG", style="Section.TLabel").pack(
            anchor="w", padx=8, pady=(9, 5)
        )
        columns = ("index", "name", "time")
        self.pass_tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        self.pass_tree.heading("index", text="#")
        self.pass_tree.heading("name", text="Pass")
        self.pass_tree.heading("time", text="ms")
        self.pass_tree.column("index", width=38, anchor="e", stretch=False)
        self.pass_tree.column("name", width=185, stretch=False)
        self.pass_tree.column("time", width=65, anchor="e", stretch=False)
        self.pass_tree.tag_configure("modeled", foreground="#80dfcf")
        self.pass_tree.tag_configure("approximated", foreground="#e6bd6b")
        self.pass_tree.tag_configure("documented", foreground="#9aa9bf")
        tree_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.pass_tree.yview)
        self.pass_tree.configure(yscrollcommand=tree_scroll.set)
        self.pass_tree.pack(side="top", fill="both", expand=True, padx=(8, 22))
        tree_scroll.place(relx=1.0, rely=0.04, relheight=0.68, x=-10, anchor="ne")

        ttk.Label(parent, text="TILE PROBE", style="Section.TLabel").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        ttk.Label(
            parent,
            textvariable=self.inspector_var,
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).pack(anchor="w", padx=8)
        self.metrics = tk.Text(
            parent,
            height=8,
            bg=self.colors["panel_2"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.colors["accent"],
            font=("Cascadia Mono", 9),
            padx=8,
            pady=8,
        )
        self.metrics.pack(fill="x", padx=8, pady=(8, 10))
        self.metrics.insert("1.0", "No world generated")
        self.metrics.configure(state="disabled")

    def _config(self) -> WorldConfig:
        enabled = tuple(name for name, variable in self.phase_vars.items() if variable.get())
        return WorldConfig(
            seed=self.seed_var.get() or "TerraForge",
            scale=WorldScale(self.scale_var.get()),
            evil=Evil(self.evil_var.get()),
            difficulty=Difficulty(self.difficulty_var.get()),
            hardmode=self.hardmode_var.get(),
            enabled_phases=enabled,
        )

    def randomize_seed(self) -> None:
        self.seed_var.set(f"forge-{secrets.token_hex(4)}")

    def _toggle_theme(self) -> None:
        self.colors = _LIGHT_THEME if self.light_mode_var.get() else _DARK_THEME
        self.root.configure(bg=self.colors["bg"])
        self.header.configure(bg=self.colors["bg"], highlightbackground=self.colors["accent"])
        self.footer.configure(bg=self.colors["bg"])
        self.logo_label.configure(bg=self.colors["bg"])
        self.canvas.configure(bg=self.colors["canvas"], highlightbackground=self.colors["accent"])
        self.controls_canvas.configure(
            bg=self.colors["panel"], highlightbackground=self.colors["accent"]
        )
        self.metrics.configure(
            bg=self.colors["panel_2"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            highlightbackground=self.colors["accent"],
        )
        self._configure_styles()

    def generate(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        config = self._config()
        self.cancel_event = threading.Event()
        self.progress["value"] = 0
        self.pass_tree.delete(*self.pass_tree.get_children())
        self.generate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("Heating the world forge...")

        def run() -> None:
            try:
                world = TerraForgePipeline().generate(config, self._on_progress, self.cancel_event)
                self.events.put(("done", world))
            except GenerationCancelledError as error:
                self.events.put(("cancelled", error))
            except Exception as error:  # GUI must surface worker failures.
                self.events.put(("error", error))

        self.worker = threading.Thread(target=run, name="terraforge-generator", daemon=True)
        self.worker.start()

    def _on_progress(self, event: PassEvent) -> None:
        self.events.put(("progress", event))

    def cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling after the active pass...")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self._show_progress(payload)  # type: ignore[arg-type]
                elif kind == "done":
                    self._generation_done(payload)  # type: ignore[arg-type]
                elif kind == "cancelled":
                    self._generation_stopped("Generation cancelled")
                elif kind == "error":
                    self._generation_stopped("Generation failed")
                    messagebox.showerror("TerraForge", str(payload))
        except queue.Empty:
            pass
        self.root.after(60, self._poll_events)

    def _show_progress(self, event: PassEvent) -> None:
        self.progress["value"] = event.progress * 100
        if event.finished:
            self.pass_tree.insert(
                "",
                "end",
                values=(event.spec.index, event.spec.name, f"{event.elapsed_ms:.1f}"),
                tags=(event.spec.fidelity.value,),
            )
            children = self.pass_tree.get_children()
            if children:
                self.pass_tree.see(children[-1])
        self.status_var.set(f"{event.spec.index:03d}/{event.total:03d} | {event.spec.name}")

    def _generation_done(self, world: GeneratedWorld) -> None:
        self.previous_world = self.current_world
        self.current_world = world
        self.display_scale = 4 if world.config.scale is WorldScale.PREVIEW else 1
        self.progress["value"] = 100
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.status_var.set(
            f"World sealed | {world.metadata['generation_seconds']:.2f}s | "
            f"{len(world.structures)} landmarks discovered"
        )
        self._update_metrics()
        self._render_view()

    def _generation_stopped(self, status: str) -> None:
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.status_var.set(status)

    def _world_for_view(self) -> GeneratedWorld | None:
        if self.compare_var.get() == "Previous":
            return self.previous_world or self.current_world
        return self.current_world

    def _render_view(self) -> None:
        world = self._world_for_view()
        if world is None:
            return
        image = render_world(
            world,
            self.display_scale,
            biome_overlay=self.biomes_var.get(),
            layer_lines=self.layers_var.get(),
            markers=self.markers_var.get(),
        )
        if self.compare_var.get() == "Split comparison" and self.previous_world is not None:
            previous = render_world(
                self.previous_world,
                self.display_scale,
                biome_overlay=self.biomes_var.get(),
                layer_lines=self.layers_var.get(),
                markers=self.markers_var.get(),
            )
            height = max(image.height, previous.height)
            combined = Image.new("RGB", (previous.width + image.width + 4, height), "#63d3c1")
            combined.paste(previous, (0, 0))
            combined.paste(image, (previous.width + 4, 0))
            image = combined
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def _fit(self) -> None:
        world = self._world_for_view()
        if world is None:
            return
        self.root.update_idletasks()
        available_w = max(1, self.canvas.winfo_width() - 4)
        available_h = max(1, self.canvas.winfo_height() - 4)
        self.display_scale = max(
            1, min(8, int(min(available_w / world.shape[1], available_h / world.shape[0])))
        )
        self._render_view()

    def _zoom(self, delta: int) -> None:
        self.display_scale = max(1, min(8, self.display_scale + delta))
        self._render_view()

    def _mouse_zoom(self, event: tk.Event) -> None:
        self._zoom(1 if event.delta > 0 else -1)

    def _canvas_press(self, event: tk.Event) -> None:
        self._drag_origin = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _canvas_drag(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _inspect_tile(self, event: tk.Event) -> None:
        world = self._world_for_view()
        if world is None:
            return
        if hasattr(self, "_drag_origin") and (
            abs(event.x - self._drag_origin[0]) > 4 or abs(event.y - self._drag_origin[1]) > 4
        ):
            return
        x = int(self.canvas.canvasx(event.x) / self.display_scale)
        y = int(self.canvas.canvasy(event.y) / self.display_scale)
        if not (0 <= x < world.shape[1] and 0 <= y < world.shape[0]):
            return
        tile = Tile(int(world.tiles[y, x]))
        wall = Wall(int(world.walls[y, x]))
        liquid = Liquid(int(world.liquid_kind[y, x]))
        biome = Biome(int(world.biomes[y, x]))
        self.inspector_var.set(
            f"({x}, {y})  {TILE_STYLES[tile].name}\n"
            f"Biome: {BIOME_NAMES[biome]} | Wall: {wall.name.title()}\n"
            f"Liquid: {liquid.name.title()} ({int(world.liquid_amount[y, x])}/255)"
        )

    def _update_metrics(self) -> None:
        world = self.current_world
        if world is None:
            return
        text = (
            f"seed       {world.metadata['seed_value']}\n"
            f"world      {world.shape[1]} x {world.shape[0]}\n"
            f"time       {world.metadata['generation_seconds']:.3f} s\n"
            f"memory     {world.memory_bytes / 1024 / 1024:.2f} MiB\n"
            f"air        {world.metadata['air_fraction'] * 100:.1f}%\n"
            f"structures {len(world.structures)}\n"
            f"evil       {world.config.evil.value}\n"
            f"depth      {world.layers.underworld} to Underworld"
        )
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", "end")
        self.metrics.insert("1.0", text)
        self.metrics.configure(state="disabled")

    def export_png(self) -> None:
        if self.current_world is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG image", "*.png")]
        )
        if path:
            save_png(self.current_world, path, self.display_scale, markers=True)

    def export_gif(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".gif", filetypes=[("GIF animation", "*.gif")]
        )
        if not path:
            return
        config = self._config()
        config = WorldConfig(
            seed=config.seed,
            scale=WorldScale.PREVIEW,
            evil=config.evil,
            difficulty=config.difficulty,
            hardmode=config.hardmode,
            enabled_phases=config.enabled_phases,
        )
        self.status_var.set("Rendering generation milestones...")
        try:
            save_generation_gif(config, path, scale=4)
            self.status_var.set(f"Saved {Path(path).name}")
        except Exception as error:
            messagebox.showerror("TerraForge", str(error))

    def export_npz(self) -> None:
        if self.current_world is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("NumPy archive", "*.npz")]
        )
        if path:
            save_npz(self.current_world, path)


def main() -> None:
    root = tk.Tk()
    TerraForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
