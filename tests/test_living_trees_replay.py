"""Synthetic invariants and numeric runtime vectors; no proprietary world required."""

import copy
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.fidelity_living_trees_replay import summarize_trace
from terraexplorer.fidelity.living_trees import LivingTreesReplay, UnsupportedCallError
from terraexplorer.fidelity.pass_snapshot import (
    RAW_DTYPE,
    canonical_cells,
    compare_cells,
    load_snapshot,
)
from terraexplorer.fidelity.unified_random import MAX_INT, MIN_INT, UnifiedRandom, int32
from terraexplorer.fidelity.wld325 import CELL, decode_tiles

VECTORS = json.loads((Path(__file__).parent / "fixtures/terraria1457_rng.json").read_text())


def state_hash(rng):
    data = json.dumps(rng.state(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: str(case["seed"]))
def test_actual_pinned_runtime_rng_vectors(case):
    rng = UnifiedRandom(case["seed"])
    assert state_hash(rng) == case["initial_state_sha256"]
    for operation in case["operations"]:
        method = {"Next": rng.next, "NextDouble": rng.next_double, "Peek": rng.peek}[
            operation["method"]
        ]
        value = method(*operation["args"])
        assert value == operation["value"]
        if operation["double_bits"] is not None:
            assert struct.pack(">d", value).hex() == operation["double_bits"]
        assert state_hash(rng) == operation["state_sha256"]
    for _ in range(1000):
        rng.next()
    assert state_hash(rng) == case["after_1000_more_next_sha256"]


def test_rng_invalid_ranges_do_not_advance_and_state_has_no_aliases():
    rng = UnifiedRandom(123)
    original = rng.state()
    for args in ((-1,), (4, 3), (MIN_INT - 1, MAX_INT), (1, 2, 3)):
        with pytest.raises(ValueError):
            rng.next(*args)
        assert rng.state() == original
    clone = UnifiedRandom.from_state(original)
    clone.next()
    assert rng.state() == original
    original["seed_array"][1] = 0
    assert rng.state()["seed_array"][1] != 0
    with pytest.raises(ValueError):
        UnifiedRandom.from_state({"inext": 56, "seed_array": [0] * 56})
    assert int32(MAX_INT + 1) == MIN_INT
    assert int32(MIN_INT - 1) == MAX_INT


def test_native_projection_matches_independently_specified_saved_wire_bytes():
    native = np.zeros((1, 1), dtype=RAW_DTYPE)
    native[0, 0] = (300, 513, 200, 0xBBE7, 0xE9, 0, 0xE0, 18, 36)
    importance = [False] * 301
    importance[300] = True
    # The v325 test wire independently encodes framed/painted tile 300, wall
    # 513, shimmer, slope 3, all wires, actuator/inactive and four coatings.
    encoded = bytes([0x3F, 0x4F, 0xFF, 0x1E]) + struct.pack(
        "<HhhBBBBB", 300, 18, 36, 7, 1, 9, 200, 2
    )
    decoded, _ = decode_tiles(encoded, 1, 1, importance)
    assert canonical_cells(native, importance).tobytes() == decoded
    assert decoded == CELL.pack(1, 300, 513, 18, 36, 7, 9, 4, 200, 4, 63, 30)
    native["sTileHeader"] &= np.uint16(0xFFDF)
    projected = canonical_cells(native, importance)
    assert projected["tile"][0, 0] == 0 and projected["frame_x"][0, 0] == -1
    assert native["type"][0, 0] == 300 and native["frameX"][0, 0] == 18


def test_diff_separates_missing_extra_and_wrong_field_writes():
    pre = np.zeros((3, 2), dtype=RAW_DTYPE)
    expected, actual = pre.copy(), pre.copy()
    expected["type"][0, 0] = 191
    expected["wall"][1, 1] = 244
    actual["wall"][1, 1] = 78
    actual["liquid"][2, 0] = 8
    diff = compare_cells(pre, expected, actual)
    assert diff["expected_changed_cells"] == diff["actual_changed_cells"] == 2
    assert diff["missing_changed_cells"] == diff["extra_changed_cells"] == 1
    assert diff["different_cells"] == 3
    assert diff["per_field_differences"]["wall"] == 1
    assert diff["first_differences"][0] == {
        "x": 0,
        "y": 0,
        "fields": {"type": {"expected": 191, "actual": 0}},
    }


def test_truncated_snapshot_and_unbalanced_trace_fail_closed(tmp_path):
    (tmp_path / "pre.json").write_text(json.dumps({"width": 3, "height": 2}))
    (tmp_path / "pre.raw.gz").write_bytes(gzip.compress(b"\0" * 14))
    with pytest.raises(ValueError, match="Incomplete"):
        load_snapshot(tmp_path, "pre")
    with pytest.raises(ValueError, match="Incomplete invocation"):
        summarize_trace([{"event": "enter", "id": 1, "parent": 0, "method": "GrowLivingTree"}])


def synthetic(seed=3):
    cells = np.zeros((600, 400), dtype=RAW_DTYPE)
    cells["sTileHeader"][:, 200:] = 32
    solid, clouds, ore = [False] * 800, [False] * 800, [False] * 800
    for kind in (0, 1, 2, 40, 48, 191, 192):
        solid[kind] = True
    clouds[189] = True
    data = dict.fromkeys(
        (
            "DrunkWorld",
            "NotTheBees",
            "ForTheWorthy",
            "Anniversary",
            "DontStarve",
            "RemixWorld",
            "NoTrapsWorld",
            "ZenithWorld",
            "SkyblockWorld",
        ),
        False,
    )
    data.update(_seed=seed, _seedText=str(seed), GameMode=0, HasCorruption=True, IsHardMode=False)
    metadata = {
        "width": 600,
        "height": 400,
        "rng": UnifiedRandom(seed).state(),
        "world_file_data": data,
        "secret_seeds": {"extraLivingTrees": False},
        "globals": {
            "Terraria.Main": {
                "worldSurface": 250.0,
                "tileSolid": solid,
                "tileSolidTop": [False] * 800,
                "wallDungeon": [False] * 500,
            },
            "Terraria.WorldGen": {
                **dict.fromkeys(
                    (
                        "drunkWorldGen",
                        "notTheBees",
                        "tenthAnniversaryWorldGen",
                        "remixWorldGen",
                        "skyblockWorldGen",
                    ),
                    False,
                ),
                "beachDistance": 60,
            },
            "Terraria.WorldBuilding.GenVars": {"mCaveX": [], "numMCaves": 0},
            "Terraria.ID.TileID+Sets": {"Ore": ore, "Clouds": clouds},
        },
    }
    return cells, metadata


def test_zero_tree_pass_preserves_absence_and_consumes_selection_draws():
    cells, meta = synthetic()
    replay = LivingTreesReplay(cells, meta)
    expected_rng = UnifiedRandom(3)
    assert expected_rng.next(0, 1) == 0 and expected_rng.next(2) == 1
    replay.run()
    assert replay.trace == [] and replay.candidates == []
    assert np.array_equal(cells, replay.cells)
    assert replay.rng.state() == expected_rng.state()


@pytest.mark.parametrize("header", [0, 32 | 64, 32 | 1024, 32 | 4096])
def test_invalid_support_rejects_without_rng_or_mutation(header):
    cells, meta = synthetic()
    cells["sTileHeader"][120, 200] = header
    replay = LivingTreesReplay(cells, meta)
    assert replay.grow(120, 199) is False
    assert replay.rng.state() == meta["rng"]
    assert np.array_equal(cells, replay.cells)


def test_clearance_rejection_preserves_draws_and_patch_semantics():
    cells, meta = synthetic()
    cells["sTileHeader"][110, 180] = 32
    cells["type"][110, 180] = 41
    normal, patch = LivingTreesReplay(cells, meta), LivingTreesReplay(cells, meta)
    assert normal.grow(120, 199) is False
    assert patch.grow(120, 199, True) is False
    assert normal.rng.state() != meta["rng"]
    continued = UnifiedRandom.from_state(normal.rng.state())
    continued.next(1, 3)
    continued.next(1, 3)
    assert patch.rng.state() == continued.state()
    assert np.array_equal(normal.cells, cells) and np.array_equal(patch.cells, cells)


def test_leaf_predicate_respects_presence_clouds_trunk_walls_and_bounds():
    cells, meta = synthetic()
    replay = LivingTreesReplay(cells, meta)
    replay.cells["type"][100, 100] = 189
    assert replay.can_place_leaves(100, 100)  # Latent inactive cloud ID is allowed.
    replay.cells["sTileHeader"][100, 100] = 32
    assert not replay.can_place_leaves(100, 100)
    replay.cells["type"][100, 100] = 0
    assert replay.can_place_leaves(100, 100)
    replay.cells["type"][100, 100] = 191
    assert not replay.can_place_leaves(100, 100)
    replay.cells["sTileHeader"][100, 100] = 0
    replay.cells["wall"][100, 100] = 244
    assert not replay.can_place_leaves(100, 100)
    assert not replay.can_place_leaves(4, 100)


def test_geometry_stops_at_dependency_without_fabricating_a_result():
    class PrefixReplay(LivingTreesReplay):
        def unsupported(self, method, *args):
            raise UnsupportedCallError(method, args, self.rng)

    cells, meta = synthetic(314159265)
    original = cells.copy()
    replay = PrefixReplay(cells, meta)
    with pytest.raises(UnsupportedCallError) as caught:
        replay.grow(120, 199)
    assert caught.value.method in ("PlaceTile", "PlaceSmallPile")
    assert caught.value.rng == replay.rng.state()
    assert "result" not in replay.trace[0]
    assert np.count_nonzero(replay.cells != cells) > 0
    assert np.array_equal(cells, original)
    changed = replay.cells[replay.cells != cells]
    assert set(changed["type"]) <= {191, 192}


def test_secret_seed_rejected_before_running():
    cells, meta = synthetic()
    altered = copy.deepcopy(meta)
    altered["secret_seeds"]["extraLivingTrees"] = True
    with pytest.raises(ValueError, match="secret-seed"):
        LivingTreesReplay(cells, altered)
