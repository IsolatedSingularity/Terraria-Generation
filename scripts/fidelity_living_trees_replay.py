"""Validate local LivingTrees captures and report the independent replay boundary.

Exit 0 means a full match; 2 means an incomplete replay at a verified exact
boundary; 1 means divergence or an unverified boundary. Runtime data stays in audit/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from terraexplorer.fidelity.living_trees import LivingTreesReplay, UnsupportedCallError
from terraexplorer.fidelity.pass_snapshot import (
    array_hash,
    canonical_cells,
    compare_cells,
    load_snapshot,
)
from terraexplorer.fidelity.unified_random import UnifiedRandom
from terraexplorer.fidelity.wld325 import compare_worlds, read_world

ROOT = Path(__file__).resolve().parents[1]


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_trace(events):
    stack, calls = [], {}
    for event in events:
        if event["event"] == "enter":
            if event["id"] in calls or event["parent"] != (stack[-1] if stack else 0):
                raise ValueError("Invalid invocation nesting/identity")
            calls[event["id"]] = dict(event)
            stack.append(event["id"])
        elif event["event"] == "exit":
            if not stack or stack.pop() != event["id"]:
                raise ValueError("Unbalanced invocation trace")
            call = calls[event["id"]]
            if call["method"] != event["method"]:
                raise ValueError("Invocation method changed")
            call.update(result=event["result"], args_after=event["args"], rng_after=event["rng"])
        elif event["event"] not in ("pass_before", "pass_after"):
            raise ValueError("Unknown capture event")
    if stack:
        raise ValueError("Incomplete invocation trace")
    return list(calls.values())


def replay(capture: Path, control_world: Path, captured_world: Path, output: Path):
    audit = (ROOT / "audit").resolve()
    for path in (capture, output):
        if not path.resolve().is_relative_to(audit):
            raise ValueError("Runtime snapshots and replay output must remain under local audit/")
    if output.exists():
        raise FileExistsError(output)
    if any((capture / name).exists() for name in ("instrumentation-error.txt", "host-error.txt")):
        raise ValueError("Instrumentation reported an error")
    lock = json.loads((ROOT / "docs/fidelity/TARGET_LOCK.json").read_text())
    corpus = ROOT / lock["corpus_root_relative"]
    pins = {
        name: file_hash(corpus / name) == sha
        for name, sha in {**lock["hashes"], **lock["source_sha256"]}.items()
    }
    pins.update(
        {
            row["path"]: file_hash(ROOT / row["path"]) == row["sha256"]
            for row in lock["artifacts"].values()
        }
    )
    if not all(pins.values()):
        raise ValueError("Pinned reference changed")
    control, generated = read_world(control_world), read_world(captured_world)
    comparison = compare_worlds(control, generated)
    if not comparison["semantic_equal_under_declared_normalization"]:
        raise ValueError("Instrumented saved world differs from control; capture is invalid")
    world_metadata = generated[0]["metadata"]
    if (
        world_metadata["width"],
        world_metadata["height"],
        world_metadata["game_mode"],
        world_metadata["crimson"],
        world_metadata["hardmode"],
    ) != (4200, 1200, 0, False, False):
        raise ValueError("Expected Small Classic Corruption pre-Hardmode saved world")
    pre, before = load_snapshot(capture, "pre")
    post, after = load_snapshot(capture, "post")
    if before["pass"] != "Living Trees" or after["pass"] != before["pass"]:
        raise ValueError("Wrong pass")
    if str(before["world_file_data"]["_seed"]) != world_metadata["seed"]:
        raise ValueError("Captured and saved seeds disagree")
    events = [json.loads(line) for line in (capture / "calls.jsonl").read_text().splitlines()]
    calls = summarize_trace(events)
    if events[0] != {"event": "pass_before", "rng": before["rng"]} or events[-1] != {
        "event": "pass_after",
        "rng": after["rng"],
    }:
        raise ValueError("Pass snapshot RNG disagrees with trace")
    manifest = next(
        row for row in generated[0]["manifest"]["GenPassResults"] if row["Name"] == "Living Trees"
    )
    if json.loads((capture / "pass-result.json").read_text()) != manifest:
        raise ValueError("Captured manifest row disagrees with genuine saved file")
    manifest_rng = UnifiedRandom.from_state(after["rng"])
    rand_next = manifest_rng.next()
    if rand_next != manifest["RandNext"]:
        raise ValueError("Post-application RNG fails the actual manifest draw")
    implementation = LivingTreesReplay(pre, before)
    stopped = None
    try:
        implementation.run()
    except UnsupportedCallError as exc:
        stopped = {"method": exc.method, "args": exc.arguments, "rng": exc.rng, "reason": str(exc)}
    expected_trees = [c for c in calls if c["method"] == "GrowLivingTree"]
    prefix_comparisons = []
    for index, actual in enumerate(implementation.trace):
        expected = expected_trees[index] if index < len(expected_trees) else None
        prefix_comparisons.append(
            {
                "index": index,
                "args": actual["args"],
                "entry_equal": expected is not None
                and actual["args"] == expected["args"]
                and actual["rng_before"] == expected["rng"],
                "result_present": "result" in actual,
                "result_equal": None
                if "result" not in actual
                else expected is not None
                and actual["result"] == expected["result"]
                and actual["rng_after"] == expected["rng_after"],
            }
        )
    sequence_equal = len(implementation.trace) == len(expected_trees) and all(
        c["entry_equal"] and c["result_equal"] is True for c in prefix_comparisons
    )
    importance = before["frame_importance"]
    if importance != after["frame_importance"]:
        raise ValueError("Frame-importance table changed")
    native_diff = compare_cells(pre, post, implementation.cells)
    canonical_pre, canonical_post = (
        canonical_cells(pre, importance),
        canonical_cells(post, importance),
    )
    canonical_actual = canonical_cells(implementation.cells, importance)
    canonical_diff = compare_cells(canonical_pre, canonical_post, canonical_actual)
    checkpoint = None
    previous_checkpoint = None
    if (capture / "first-object.json").exists():

        class OriginalPrefix(LivingTreesReplay):
            def unsupported(self, method, *args):
                raise UnsupportedCallError(method, args, self.rng)

        prefix = OriginalPrefix(pre, before)
        try:
            prefix.run()
        except UnsupportedCallError as exc:
            old_cells, old_meta = load_snapshot(capture, "first-object")
            old_call = next(c for c in calls if c["method"] in ("PlaceTile", "PlaceSmallPile"))
            previous_checkpoint = {
                "native_comparison": compare_cells(pre, old_cells, prefix.cells),
                "call_equal": exc.method == old_call["method"]
                and exc.arguments == old_call["args"],
                "rng_equal": exc.rng == old_meta["rng"] == old_call["rng"],
                "tile_solid_table_equal": prefix.solid_types
                == old_meta["globals"]["Terraria.Main"]["tileSolid"],
                "chests_equal": before["chests"] == old_meta["chests"],
            }
    checkpoint_phase = (
        "first-passage"
        if stopped and stopped["method"] == "GrowLivingTree_MakePassage"
        else "first-object"
    )
    if stopped and (capture / f"{checkpoint_phase}.json").exists():
        cp, cp_meta = load_snapshot(capture, checkpoint_phase)
        boundary = next(c for c in calls if c["method"] == stopped["method"])
        expected_objects = [
            c
            for c in calls
            if c["id"] < boundary["id"] and c["method"] in ("PlaceTile", "PlaceSmallPile")
        ]
        object_sequence = [
            {
                "method": c["method"],
                "args": c["args"],
                "rng_before": c["rng"],
                "result": c["result"],
                "rng_after": c["rng_after"],
            }
            for c in expected_objects
        ]
        from scripts.fidelity_living_trees_objects import normalized_events

        nested = []
        for obj in expected_objects:
            start = next(
                i
                for i, e in enumerate(events)
                if e.get("id") == obj["id"] and e["event"] == "enter"
            )
            end = next(
                i for i, e in enumerate(events) if e.get("id") == obj["id"] and e["event"] == "exit"
            )
            nested.extend(normalized_events(events[start : end + 1]))
        actual_events = implementation.placement.events if implementation.placement else []
        global_differences = {
            t: [
                f
                for f, v in fields.items()
                if v != cp_meta["globals"][t][f]
                and not (t == "Terraria.Main" and f in ("tileSolid", "statusText", "oldStatusText"))
            ]
            for t, fields in before["globals"].items()
        }
        checkpoint = {
            "phase": checkpoint_phase,
            "native_comparison": compare_cells(pre, cp, implementation.cells),
            "call_equal": stopped["method"] == boundary["method"]
            and stopped["args"] == boundary["args"],
            "rng_equal": stopped["rng"] == cp_meta["rng"] == boundary["rng"],
            "tile_solid_table_equal": implementation.solid_types
            == cp_meta["globals"]["Terraria.Main"]["tileSolid"],
            "chests_equal": before["chests"] == cp_meta["chests"],
            "unresolved_global_differences": global_differences,
            "relevant_globals_equal": not any(global_differences.values()),
            "object_call_count": len(implementation.object_calls),
            "object_result_rng_sequence_equal": implementation.object_calls == object_sequence,
            "nested_object_event_count": len(nested),
            "nested_object_trace_equal": actual_events == nested,
            "vanilla_call_result": boundary["result"],
            "vanilla_call_rng_changed": boundary["rng"] != boundary["rng_after"],
        }
    # Exact net-change masks, not an inferred history of individual write operations.
    exports = output.parent / (output.stem + "-exports")
    exports.mkdir(parents=True, exist_ok=False)
    for phase, cells in (
        ("pre", canonical_pre),
        ("post", canonical_post),
        ("actual", canonical_actual),
    ):
        (exports / f"{phase}.canonical.gz").write_bytes(
            gzip.compress(cells.tobytes(), compresslevel=1, mtime=0)
        )
    for schema, a, z in (("native", pre, post), ("canonical", canonical_pre, canonical_post)):
        for field in ("all", *a.dtype.names):
            mask = z != a if field == "all" else z[field] != a[field]
            (exports / f"{schema}-{field}.mask.gz").write_bytes(
                gzip.compress(
                    np.packbits(mask.ravel(order="C"), bitorder="little").tobytes(), mtime=0
                )
            )
    global_changes = {
        t: [f for f, v in fields.items() if v != after["globals"][t][f]]
        for t, fields in before["globals"].items()
    }
    unresolved_globals = {
        t: [
            f
            for f in fields
            if not (
                t == "Terraria.Main" and f in ("tileSolid", "chest", "statusText", "oldStatusText")
            )
        ]
        for t, fields in global_changes.items()
    }
    declared_state_equal = (
        implementation.solid_types == after["globals"]["Terraria.Main"]["tileSolid"]
        and before["chests"] == after["chests"]
        and before["secret_seeds"] == after["secret_seeds"]
        and not any(unresolved_globals.values())
    )
    checkpoint_equal = (
        checkpoint is not None
        and all(
            checkpoint[key]
            for key in (
                "call_equal",
                "rng_equal",
                "tile_solid_table_equal",
                "chests_equal",
                "relevant_globals_equal",
                "object_result_rng_sequence_equal",
                "nested_object_trace_equal",
            )
        )
        and checkpoint["native_comparison"]["equal"]
        and all(
            c["entry_equal"] and (not c["result_present"] or c["result_equal"])
            for c in prefix_comparisons
        )
    )
    if previous_checkpoint is not None:
        checkpoint_equal = (
            checkpoint_equal
            and previous_checkpoint["native_comparison"]["equal"]
            and all(v for k, v in previous_checkpoint.items() if k != "native_comparison")
        )
    report = {
        "status": "INCOMPLETE_UNSUPPORTED_CALL"
        if stopped and checkpoint_equal
        else "MATCH"
        if native_diff["equal"]
        and sequence_equal
        and implementation.rng.state() == after["rng"]
        and declared_state_equal
        else "DIVERGED",
        "target": "Terraria 1.4.5.7 / BuildID 24825745 / world format 325",
        "seed": world_metadata["seed"],
        "world_metadata": world_metadata,
        "pin_checks": pins,
        "control_sha256": control[0]["sha256"],
        "captured_world_sha256": generated[0]["sha256"],
        "instrumentation_nonperturbing": comparison,
        "pass": manifest,
        "pass_config": json.loads((capture / "pass-config.json").read_text()),
        "secret_seed_flags": before["secret_seeds"],
        "pass_rng_initial_sha256": json_hash(before["rng"]),
        "pass_rng_final_sha256": json_hash(after["rng"]),
        "manifest_RandNext_verified": rand_next,
        "manifest_rng_state_after_draw_sha256": json_hash(manifest_rng.state()),
        "native_pre_sha256": array_hash(pre),
        "native_post_sha256": array_hash(post),
        "canonical_pre_sha256": array_hash(canonical_pre),
        "canonical_post_sha256": array_hash(canonical_post),
        "canonical_actual_sha256": array_hash(canonical_actual),
        "native_full_post_comparison": native_diff,
        "canonical_full_post_comparison": canonical_diff,
        "first_unsupported_call": stopped,
        "first_object_checkpoint": checkpoint if checkpoint_phase == "first-object" else None,
        "first_passage_checkpoint": checkpoint if checkpoint_phase == "first-passage" else None,
        "verified_exact_boundary": checkpoint_equal,
        "previous_first_object_checkpoint": previous_checkpoint,
        "candidate_x_draws_before_stop": implementation.candidates,
        "invocation_prefix_comparison": prefix_comparisons,
        "full_invocation_result_sequence_equal": sequence_equal,
        "full_pass_final_rng_equal": implementation.rng.state() == after["rng"],
        "full_pass_tile_solid_table_equal": implementation.solid_types
        == after["globals"]["Terraria.Main"]["tileSolid"],
        "chests": {
            "pre_count": len(before["chests"]),
            "post_count": len(after["chests"]),
            "replay_equal": before["chests"] == after["chests"],
        },
        "observed_global_field_changes": global_changes,
        "unresolved_global_field_changes": unresolved_globals,
        "invocation_counts": dict(Counter(c["method"] for c in calls)),
        "tree_invocations": [
            {
                "args": c["args"],
                "result": c["result"],
                "rng_before_sha256": json_hash(c["rng"]),
                "rng_after_sha256": json_hash(c["rng_after"]),
            }
            for c in expected_trees
        ],
        "capture_file_sha256": {p.name: file_hash(p) for p in capture.iterdir() if p.is_file()},
        "export_file_sha256": {p.name: file_hash(p) for p in exports.iterdir()},
        "script_sha256": file_hash(Path(__file__)),
        "comparison_scope": "All native Tile fields and saved-cell projection; relevant solidity "
        "table and chest state. Missing/extra writes are net field changes, not temporal write "
        "logs. Snapshots retain additional globals with explicit unexpanded_type markers; "
        "arbitrary runtime object graphs and progress UI are not replayed.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "seed": report["seed"],
                "status": report["status"],
                "checkpoint": checkpoint,
                "full_native": native_diff,
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "MATCH" else 2 if checkpoint_equal else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--control-world", required=True, type=Path)
    parser.add_argument("--captured-world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return replay(args.capture, args.control_world, args.captured_world, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
