from types import SimpleNamespace

from PIL import Image

from terraexplorer.config import WorldConfig, WorldScale
from terraexplorer.gui import TerraExplorerApp, _windows_colorref
from terraexplorer.model import GeneratedWorld


def test_windows_colorref_uses_dwm_byte_order() -> None:
    assert _windows_colorref("#c6a36f") == 0x6FA3C6


def test_small_fit_preserves_whole_world_and_tile_inspection(monkeypatch):
    """Exercise GUI render transforms without requiring a desktop in CI."""
    world = GeneratedWorld.empty(WorldConfig(scale=WorldScale.SMALL))
    app = TerraExplorerApp.__new__(TerraExplorerApp)
    app._world_for_view = lambda: world
    app.root = SimpleNamespace(update_idletasks=lambda: None, after_idle=lambda fn: fn())

    def noop(*a, **kw):
        pass

    app.canvas = SimpleNamespace(
        winfo_width=lambda: 800,
        winfo_height=lambda: 400,
        delete=noop,
        create_image=noop,
        configure=noop,
        xview_moveto=noop,
        yview_moveto=noop,
        canvasx=lambda x: x,
        canvasy=lambda y: y,
    )
    app.compare_var = SimpleNamespace(get=lambda: "Current")
    app.biomes_var = app.layers_var = app.markers_var = SimpleNamespace(get=lambda: False)
    app.evolution_frames = []
    app.previous_world = None
    app.display_scale = 1
    app.fit_view = False
    monkeypatch.setattr(
        "terraexplorer.gui.render_world", lambda *a, **kw: Image.new("RGB", (4200, 1200))
    )
    monkeypatch.setattr(
        "terraexplorer.gui.ImageTk.PhotoImage",
        lambda im: SimpleNamespace(width=lambda: im.width, height=lambda: im.height),
    )
    app._fit()
    assert app.photo.width() <= 796 and app.photo.height() <= 396
    assert abs(app.photo.width() / 3.5 - app.photo.height()) <= 1
    messages = []
    app.inspector_var = SimpleNamespace(set=messages.append)
    app._inspect_tile(
        SimpleNamespace(x=1000.5 * app.view_pixel_scale[0], y=500.5 * app.view_pixel_scale[1])
    )
    assert messages[0].startswith("(1000, 500)")
