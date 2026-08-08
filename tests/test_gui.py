from terraexplorer.gui import _windows_colorref


def test_windows_colorref_uses_dwm_byte_order() -> None:
    assert _windows_colorref("#c6a36f") == 0x6FA3C6
