# Contributing to TerraForge

Thanks for improving the generator. Changes are easiest to review when they
preserve determinism, keep fidelity claims explicit, and include a focused
test.

## Local setup

```bash
python -m pip install -e ".[dev,build]"
pytest
ruff check terraforge tests scripts packaging
ruff format --check terraforge tests scripts packaging
mypy terraforge
```

Use Python 3.11 or newer. The core must stay usable with only NumPy and Pillow;
large visualization dependencies belong in the `legacy` optional extra.

## Generation changes

- Add or change a handler in `terraforge/generation.py`.
- Register it in `terraforge/passes.py` with an honest `modeled`,
  `approximated`, or `documented` label.
- Consume only the RNG passed to the handler. Do not use global NumPy/Python
  random state.
- Keep tile, wall, liquid, and biome data in their separate arrays.
- Add deterministic tests for the visible invariant, not one exact screenshot.
- Update `docs/FIDELITY.md` when a pass status changes.

## Visual and branding changes

Do not add Terraria sprites, ripped tilesheets, logos, or other copyrighted
game assets. TerraForge's renderer and metal-tree identity must remain
original. Run `python -m scripts.generate_media` after meaningful visual or
performance changes; only commit media that the README uses.

## Pull requests

Keep a pull request scoped to one coherent change. Explain behavioral impact,
fidelity status changes, benchmark impact, and the validation commands run.
CI must pass on Windows and Ubuntu before release packaging can run.
