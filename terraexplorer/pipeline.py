"""Deterministic execution of the complete TerraExplorer pass catalogue."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from terraexplorer.config import WorldConfig
from terraexplorer.generation import PASS_HANDLERS, apply_hardmode, final_cleanup
from terraexplorer.model import GeneratedWorld, PassResult
from terraexplorer.passes import PASS_SPECS, PassSpec, Phase


class GenerationCancelledError(RuntimeError):
    """Raised when a caller cancels a running pipeline."""


@dataclass(frozen=True, slots=True)
class PassEvent:
    spec: PassSpec
    total: int
    completed: int
    progress: float
    finished: bool
    elapsed_ms: float = 0.0


ProgressCallback = Callable[[PassEvent], None]
SnapshotCallback = Callable[[PassSpec, GeneratedWorld], None]


def _rng_for(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.blake2s(f"{seed}:{label}".encode(), digest_size=8).digest()
    child_seed = int.from_bytes(digest, "little")
    return np.random.default_rng(child_seed)


class TerraExplorerPipeline:
    """Runs 107 named vanilla-order passes with per-pass RNG isolation."""

    def __init__(self, specs: tuple[PassSpec, ...] = PASS_SPECS) -> None:
        self.specs = specs

    def generate(
        self,
        config: WorldConfig,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
        snapshot: SnapshotCallback | None = None,
    ) -> GeneratedWorld:
        world = GeneratedWorld.empty(config)
        started = time.perf_counter()
        total = len(self.specs) + int(config.hardmode)

        for completed, spec in enumerate(self.specs):
            if cancel is not None and cancel.is_set():
                raise GenerationCancelledError("World generation was cancelled")
            if progress is not None:
                progress(PassEvent(spec, total, completed, completed / total, False))

            pass_started = time.perf_counter()
            phase_enabled = (
                spec.phase is Phase.TERRAIN
                or config.enabled_phases is None
                or spec.phase.value in config.enabled_phases
            )
            handler = PASS_HANDLERS.get(spec.handler or "", None) if phase_enabled else None
            if handler is not None:
                handler(world, _rng_for(config.seed_value, spec.name))
            elapsed_ms = (time.perf_counter() - pass_started) * 1000.0
            world.pass_results.append(
                PassResult(
                    index=spec.index,
                    name=spec.name,
                    phase=spec.phase.value,
                    fidelity=spec.fidelity.value,
                    elapsed_ms=elapsed_ms,
                    note=(
                        "Pass retained for order/documentation; no distinct grid mutation."
                        if spec.handler is None
                        else ""
                    ),
                )
            )
            if not phase_enabled:
                world.pass_results[-1].note = "Pass disabled by the selected phase controls."
            if snapshot is not None:
                snapshot(spec, world)
            if progress is not None:
                progress(
                    PassEvent(
                        spec,
                        total,
                        completed + 1,
                        (completed + 1) / total,
                        True,
                        elapsed_ms,
                    )
                )

        if config.hardmode:
            if cancel is not None and cancel.is_set():
                raise GenerationCancelledError("World generation was cancelled")
            hardmode_spec = PassSpec(
                len(self.specs) + 1,
                "Hardmode V Transformation",
                self.specs[-1].phase,
                self.specs[0].fidelity,
                "hardmode",
                "Post-generation event; not one of the vanilla creation passes.",
            )
            if progress is not None:
                progress(
                    PassEvent(
                        hardmode_spec,
                        total,
                        len(self.specs),
                        len(self.specs) / total,
                        False,
                    )
                )
            hardmode_started = time.perf_counter()
            apply_hardmode(world, _rng_for(config.seed_value, "Hardmode V Transformation"))
            final_cleanup(world, _rng_for(config.seed_value, "Hardmode Final Cleanup"))
            elapsed_ms = (time.perf_counter() - hardmode_started) * 1000.0
            world.pass_results.append(
                PassResult(
                    hardmode_spec.index,
                    hardmode_spec.name,
                    hardmode_spec.phase.value,
                    "modeled",
                    elapsed_ms,
                    note=hardmode_spec.note,
                )
            )
            if snapshot is not None:
                snapshot(hardmode_spec, world)
            if progress is not None:
                progress(PassEvent(hardmode_spec, total, total, 1.0, True, elapsed_ms))

        world.metadata["generation_seconds"] = time.perf_counter() - started
        world.metadata["pass_count"] = len(self.specs)
        world.metadata["executed_pass_count"] = len(world.pass_results)
        world.metadata["modeled_passes"] = sum(
            result.fidelity == "modeled" for result in world.pass_results
        )
        world.metadata["approximated_passes"] = sum(
            result.fidelity == "approximated" for result in world.pass_results
        )
        world.metadata["documented_passes"] = sum(
            result.fidelity == "documented" for result in world.pass_results
        )
        return world


def generate_world(
    config: WorldConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    snapshot: SnapshotCallback | None = None,
) -> GeneratedWorld:
    return TerraExplorerPipeline().generate(config or WorldConfig(), progress, cancel, snapshot)
