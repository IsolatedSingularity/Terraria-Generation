# PyInstaller specification for the lightweight Windows desktop build.
from pathlib import Path

root = Path(SPECPATH).parent
package_assets = root / "terraforge" / "assets"
icon = package_assets / "terraforge.ico"

a = Analysis(
    [str(root / "terraforge" / "gui.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(package_assets), "terraforge/assets")],
    hiddenimports=["tkinter", "numpy"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "seaborn"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TerraForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon) if icon.exists() else None,
)
