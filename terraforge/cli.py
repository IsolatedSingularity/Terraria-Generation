"""Compatibility wrapper for :mod:`terraexplorer.cli`."""

from terraexplorer.cli import *  # noqa: F403
from terraexplorer.cli import main

if __name__ == "__main__":
    main()
