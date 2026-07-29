"""Command-line interface shared by source installs and packaged builds."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from terraexplorer.config import Difficulty, Evil, WorldConfig, WorldScale
from terraexplorer.passes import PASS_SPECS
from terraexplorer.pipeline import PassEvent, generate_world
from terraexplorer.render import save_generation_gif, save_npz, save_png, summary_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terraexplorer",
        description="Educational, deterministic 2D world-generation laboratory.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="Generate and export a world")
    generate.add_argument("--seed", default="TerraExplorer")
    generate.add_argument("--scale", choices=[item.value for item in WorldScale], default="preview")
    generate.add_argument("--evil", choices=[item.value for item in Evil], default="corruption")
    generate.add_argument(
        "--difficulty",
        choices=[item.value for item in Difficulty],
        default="classic",
    )
    generate.add_argument("--hardmode", action="store_true")
    generate.add_argument("--png", type=Path, help="PNG output path")
    generate.add_argument("--npz", type=Path, help="Compressed NumPy output path")
    generate.add_argument("--json", type=Path, help="Pass/metadata JSON output path")
    generate.add_argument("--gif", type=Path, help="Preview generation GIF output path")
    generate.add_argument("--quiet", action="store_true")

    passes = subcommands.add_parser("passes", help="Print the complete fidelity catalogue")
    passes.add_argument("--json", action="store_true", dest="as_json")

    benchmark = subcommands.add_parser("benchmark", help="Measure generation performance")
    benchmark.add_argument(
        "--scale", choices=[item.value for item in WorldScale], default="preview"
    )
    benchmark.add_argument("--iterations", type=int, default=5)

    subcommands.add_parser("gui", help="Launch the native desktop app")
    return parser


def _config(args: argparse.Namespace) -> WorldConfig:
    return WorldConfig(
        seed=args.seed,
        scale=WorldScale(args.scale),
        evil=Evil(args.evil),
        difficulty=Difficulty(args.difficulty),
        hardmode=args.hardmode,
    )


def _progress(event: PassEvent) -> None:
    if event.finished:
        print(
            f"\r{event.completed:3d}/{event.total:3d} "
            f"{event.spec.name:<36} {event.elapsed_ms:7.1f} ms",
            end="",
            flush=True,
        )


def _generate(args: argparse.Namespace) -> int:
    config = _config(args)
    world = generate_world(config, None if args.quiet else _progress)
    if not args.quiet:
        print()
    png_path = args.png or Path("Plots/terraexplorer_world.png")
    save_png(world, png_path, 4 if config.scale is WorldScale.PREVIEW else 1, markers=True)
    if args.npz:
        save_npz(world, args.npz)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(summary_json(world), encoding="utf-8")
    if args.gif:
        preview_config = WorldConfig(
            config.seed,
            WorldScale.PREVIEW,
            config.evil,
            config.difficulty,
            config.hardmode,
        )
        save_generation_gif(preview_config, args.gif)
    print(
        f"Generated {world.shape[1]}x{world.shape[0]} in "
        f"{world.metadata['generation_seconds']:.3f}s -> {png_path}"
    )
    return 0


def _passes(args: argparse.Namespace) -> int:
    if args.as_json:
        import json

        print(
            json.dumps(
                [
                    {
                        "index": spec.index,
                        "name": spec.name,
                        "phase": spec.phase.value,
                        "fidelity": spec.fidelity.value,
                    }
                    for spec in PASS_SPECS
                ],
                indent=2,
            )
        )
        return 0
    for spec in PASS_SPECS:
        print(f"{spec.index:03d}  {spec.name:<38} {spec.fidelity.value:<12} {spec.phase.value}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    iterations = max(1, args.iterations)
    timings: list[float] = []
    for index in range(iterations):
        started = time.perf_counter()
        generate_world(WorldConfig(seed=f"benchmark-{index}", scale=WorldScale(args.scale)))
        timings.append(time.perf_counter() - started)
    print(
        f"{args.scale}: median={statistics.median(timings):.3f}s "
        f"min={min(timings):.3f}s max={max(timings):.3f}s n={iterations}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "passes":
        return _passes(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "gui":
        from terraexplorer.gui import main as gui_main

        gui_main()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
