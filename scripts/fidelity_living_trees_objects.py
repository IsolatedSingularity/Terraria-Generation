"""Replay every locally captured LivingTrees object call, independently.

Inputs and detailed output contain native runtime evidence and must stay in audit/.
Exit zero requires all calls, all native fields, globals, results and nested RNG
traces to match, as well as a semantically identical fresh control world.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.fidelity_living_trees_replay import ROOT, file_hash, summarize_trace
from terraexplorer.fidelity.living_trees import UnsupportedCallError
from terraexplorer.fidelity.object_placement import ObjectPlacement
from terraexplorer.fidelity.pass_snapshot import RAW_DTYPE, compare_cells
from terraexplorer.fidelity.unified_random import UnifiedRandom
from terraexplorer.fidelity.wld325 import compare_worlds, read_world


def normalized_events(events):
    """Keep every event/argument/result/RNG state; normalize only IDs and Tile views."""
    depth = 0
    result = []
    for event in events:
        if event["event"] == "exit":
            depth -= 1
        row = {k: event[k] for k in ("event", "method", "args", "rng")}
        row["args"] = [
            {name: arg[name] for name in RAW_DTYPE.names}
            if isinstance(arg, dict) and "sTileHeader" in arg
            else arg
            for arg in row["args"]
        ]
        row["depth"] = depth
        if event["event"] == "exit":
            row["result"] = event["result"]
        else:
            depth += 1
        result.append(row)
    if depth != 0:
        raise ValueError("Unbalanced per-object trace")
    return result


def first_difference(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        return {
            "index": min(len(expected), len(actual)),
            "expected_length": len(expected),
            "actual_length": len(actual),
        }
    return None


def validate_calls(capture):
    events = [json.loads(line) for line in (capture / "calls.jsonl").read_text().splitlines()]
    calls = summarize_trace(events)
    targets = [
        c
        for c in calls
        if c["method"] == "PlaceTile"
        and c["args"][2] == 187
        or c["method"] == "PlaceSmallPile"
        and c["args"][4] == 185
    ]
    if not targets:
        raise ValueError("No observed object calls")
    if {int(p.name) for p in (capture / "objects").iterdir()} != {c["id"] for c in targets}:
        raise ValueError("Object captures are not an exact cover of the target calls")
    offsets = {(e["id"], e["event"]): i for i, e in enumerate(events) if "id" in e}
    state_cache = {}

    def state(sha):
        path = capture / "object-states" / (sha + ".json.gz")
        if sha not in state_cache:
            data = gzip.decompress(path.read_bytes())
            import hashlib

            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Object state digest mismatch")
            state_cache[sha] = json.loads(data)
        return state_cache[sha]

    results, helpers = [], Counter()
    for call in targets:
        folder = capture / "objects" / f"{call['id']:06d}"
        meta = json.loads((folder / "capture.json").read_text())
        pre, post = [
            np.frombuffer(
                gzip.decompress((folder / f"{phase}.raw.gz").read_bytes()), dtype=RAW_DTYPE
            ).reshape(meta["shape"])
            for phase in ("pre", "post")
        ]
        changed = np.argwhere(pre != post)
        indices = [
            (int(x) + meta["origin"][0]) * meta["world_shape"][1] + int(y) + meta["origin"][1]
            for x, y in changed
        ]
        if indices != meta["changed_cell_indices"]:
            raise ValueError("Local region fails full-grid changed-cell coverage")
        # Each full-grid scan record contains absolute x-major index and both
        # native 14-byte values. Verify all records against the padded region.
        records = (folder / "changes.bin").read_bytes()
        if len(records) != 32 * len(changed):
            raise ValueError("Changed-cell record count mismatch")
        for n, (x, y) in enumerate(changed):
            record = records[n * 32 : (n + 1) * 32]
            if (
                struct.unpack_from("<i", record)[0] != indices[n]
                or record[4:18] != pre[x, y].tobytes()
                or record[18:] != post[x, y].tobytes()
            ):
                raise ValueError("Changed-cell record mismatch")
        actual = ObjectPlacement(
            pre.copy(),
            UnifiedRandom.from_state(call["rng"]),
            state(meta["pre_state"]),
            meta["origin"],
            meta["world_shape"],
        )
        result, stopped = None, None
        try:
            result = actual.invoke(call["method"], *call["args"])
        except UnsupportedCallError as exc:
            stopped = str(exc)
        expected_events = normalized_events(
            events[offsets[call["id"], "enter"] : offsets[call["id"], "exit"] + 1]
        )
        helpers.update(e["method"] for e in expected_events if e["event"] == "enter")
        native = compare_cells(pre, post, actual.cells)
        divergence = first_difference(expected_events, actual.events)
        state_after = state(meta["post_state"])
        state_differences = [k for k in state_after if state_after[k] != actual.state[k]]
        equal = (
            stopped is None
            and native["equal"]
            and result == call["result"]
            and actual.rng.state() == call["rng_after"]
            and not state_differences
            and divergence is None
        )
        results.append(
            {
                "id": call["id"],
                "method": call["method"],
                "args": call["args"],
                "equal": equal,
                "unsupported": stopped,
                "native": native,
                "return_value": result,
                "return_equal": result == call["result"],
                "rng_equal": actual.rng.state() == call["rng_after"],
                "vanilla_consumed_rng": call["rng"] != call["rng_after"],
                "state_equal": not state_differences,
                "state_differences": state_differences,
                "nested_event_count": len(expected_events),
                "nested_trace_equal": divergence is None,
                "first_trace_difference": divergence,
                "region_origin": meta["origin"],
                "region_shape": meta["shape"],
                "full_grid_change_records_verified": True,
            }
        )
    return {
        "all_equal": all(r["equal"] for r in results),
        "call_count": len(results),
        "calls": results,
        "observed_helpers": dict(helpers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "control-world", "captured-world", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    for path in (args.capture, args.control_world, args.captured_world, args.output):
        if not path.resolve().is_relative_to((ROOT / "audit").resolve()):
            raise ValueError("Runtime evidence must remain in audit/")
    if args.output.exists():
        raise FileExistsError(args.output)
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
    if any(
        (args.capture / name).exists() for name in ("instrumentation-error.txt", "host-error.txt")
    ):
        raise ValueError("Instrumentation reported an error")
    control, captured = read_world(args.control_world), read_world(args.captured_world)
    comparison = compare_worlds(control, captured)
    if not comparison["semantic_equal_under_declared_normalization"]:
        raise ValueError("Instrumented saved world differs from fresh control")
    report = validate_calls(args.capture)
    report.update(
        pin_checks=pins,
        instrumentation_nonperturbing=comparison,
        control_sha256=control[0]["sha256"],
        captured_sha256=captured[0]["sha256"],
        capture_hashes={
            str(p.relative_to(args.capture)): file_hash(p)
            for p in args.capture.rglob("*")
            if p.is_file()
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("all_equal", "call_count", "observed_helpers")}, indent=2
        )
    )
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
