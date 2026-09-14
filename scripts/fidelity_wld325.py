"""Compatibility command for the promoted, single v325 decoder."""

from terraexplorer.fidelity.wld325 import (  # noqa: F401
    CELL,
    CELL_FIELDS,
    SECTIONS,
    Reader,
    compare_worlds,
    decode_tiles,
    digest,
    main,
    read_manifest,
    read_world,
)

if __name__ == "__main__":
    main()
