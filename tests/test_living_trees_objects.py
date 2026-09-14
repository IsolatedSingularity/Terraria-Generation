"""Synthetic object geometry and header invariants; no Terraria assets or captures."""

import copy

import numpy as np
import pytest

from scripts.fidelity_living_trees_objects import first_difference, normalized_events
from terraexplorer.fidelity.living_trees import LivingTreesReplay, UnsupportedCallError
from terraexplorer.fidelity.object_placement import ObjectPlacement
from terraexplorer.fidelity.pass_snapshot import RAW_DTYPE
from terraexplorer.fidelity.unified_random import UnifiedRandom


def fixture():
    cells = np.zeros((80, 80), dtype=RAW_DTYPE)
    cells["sTileHeader"][:, 41:] = 32
    solid = [False] * 800
    for kind in (0, 1, 191, 192):
        solid[kind] = True
    important = [False] * 800
    important[185] = important[187] = True
    state = {
        "globals": {
            "Terraria.Main": {
                "tileSolid": solid,
                "tileSolidTop": [False] * 800,
                "tileFrameImportant": important,
                "dedServ": True,
                "gameMenu": True,
            },
            "Terraria.WorldGen": {"generatingWorld": True, "destroyObject": False},
            "Terraria.ID.TileID+Sets": {
                name: [False] * 800
                for name in (
                    "Platforms",
                    "Boulders",
                    "TruncatesWalls",
                    "ResetsHalfBrickPlacementAttempt",
                    "IsVine",
                )
            },
        },
        "chests": [],
        "entities": {"player": [], "npc": []},
    }
    return ObjectPlacement(cells, UnifiedRandom(17), state)


@pytest.mark.parametrize("style", range(47, 52))
def test_three_by_two_uses_style_and_anchor_and_preserves_non_object_fields(style):
    obj = fixture()
    obj.cells["wall"] = 22
    obj.cells["liquid"] = 9
    obj.cells["bTileHeader"] = 0x89
    obj.cells["sTileHeader"][:, :41] = 128  # Red wire, inactive tile.
    before_rng, before_state = obj.rng.state(), copy.deepcopy(obj.state)
    assert obj.invoke("PlaceTile", 30, 40, 187, True, False, -1, style) is True
    for x in range(29, 32):
        for y in range(39, 41):
            tile = obj.cells[x, y]
            assert (tile["type"], tile["frameX"], tile["frameY"]) == (
                187,
                style * 54 + (x - 29) * 18,
                (y - 39) * 18,
            )
            assert tile["sTileHeader"] == 160
    assert np.all(obj.cells["wall"] == 22) and np.all(obj.cells["liquid"] == 9)
    assert np.all(obj.cells["bTileHeader"] == 0x89)
    assert obj.rng.state() == before_rng and obj.state == before_state
    assert sum(e["method"] == "SquareTileFrame" and e["event"] == "enter" for e in obj.events) == 2


def test_rejected_placement_still_clears_latent_origin_and_neighbor_shape():
    obj = fixture()
    obj.cells["sTileHeader"][29, 41] = 0  # Missing one support rejects the object.
    obj.cells[30, 40] = (99, 3, 7, 0xF49F, 0xFF, 0xA5, 0xFF, 120, 240)
    obj.cells["sTileHeader"][29, 39] = 1024 | 4096 | 31 | 128
    assert obj.invoke("PlaceTile", 30, 40, 187, True, False, -1, 49) is False
    tile = obj.cells[30, 40]
    assert tile["type"] == 0 and tile["frameX"] == tile["frameY"] == 0
    assert tile["sTileHeader"] == 0xF49F & ~(31 | 1024 | 28672 | 32)
    assert tile["bTileHeader3"] == 0x5F
    assert (tile["wall"], tile["liquid"], tile["bTileHeader"], tile["bTileHeader2"]) == (
        3,
        7,
        255,
        165,
    )
    assert obj.cells["sTileHeader"][29, 39] == 128


def test_failed_object_attempt_can_return_true_on_existing_active_origin():
    obj = fixture()
    obj.cells["sTileHeader"][30, 40] = 32
    obj.cells["type"][30, 40] = 191
    assert obj.invoke("PlaceTile", 30, 40, 187, True, False, -1, 47) is True
    assert obj.cells["type"][30, 40] == 191
    assert not np.any(obj.cells["type"] == 187)


def test_neighbor_destruction_reaches_beyond_attempted_footprint():
    obj = fixture()
    assert obj.invoke("PlaceTile", 35, 40, 187, True, False, -1, 50)
    obj.cells["type"][34, 40] = 191  # A later trunk write damages the neighboring object.
    obj.cells["wall"][34:37, 39:41] = 12
    obj.cells["liquid"][34:37, 39:41] = 25
    rng = obj.rng.state()
    assert obj.invoke("PlaceTile", 33, 39, 187, True, False, -1, 49) is False
    assert not np.any(obj.cells["type"] == 187)
    assert obj.cells["type"][34, 40] == 191
    assert obj.cells["frameX"][36, 40] == obj.cells["frameY"][36, 40] == -1
    assert np.all(obj.cells["wall"][34:37, 39:41] == 12)
    assert np.all(obj.cells["liquid"][34:37, 39:41] == 25)
    assert obj.rng.state() == rng and obj.world["destroyObject"] is False
    kills = [e for e in obj.events if e["method"] == "KillTile" and e["event"] == "enter"]
    assert len(kills) == 5 and max(e["args"][0] for e in kills) == 36


@pytest.mark.parametrize("size,style", [(0, 72), (1, 59), (1, 60), (1, 61)])
def test_piles_preserve_headers_and_consume_no_rng(size, style):
    obj = fixture()
    obj.cells["sTileHeader"][30:32, 40] = 0x149F
    obj.cells["bTileHeader3"][30:32, 40] = 0xFF
    rng = obj.rng.state()
    assert obj.invoke("PlaceSmallPile", 30, 40, style, size, 185)
    for x in range(30, 31 + size):
        tile = obj.cells[x, 40]
        assert tile["type"] == 185 and tile["sTileHeader"] == 0x14BF
        assert tile["frameX"] == style * 18 * (size + 1) + (x - 30) * 18
        assert tile["frameY"] == size * 18 and tile["bTileHeader3"] == 0xFF
    assert obj.rng.state() == rng


def test_size_zero_boulder_asymmetry_lava_and_short_circuit_support():
    obj = fixture()
    obj.sets["Boulders"][0] = True
    assert obj.invoke("PlaceSmallPile", 30, 40, 72, 0, 185)
    assert obj.invoke("PlaceSmallPile", 35, 40, 59, 1, 185) is False
    obj.cells["liquid"][40, 40], obj.cells["bTileHeader"][40, 40] = 255, 32
    before = len(obj.events)
    assert obj.invoke("PlaceSmallPile", 40, 40, 72, 0, 185) is False
    assert len(obj.events) - before == 2
    obj.cells["sTileHeader"][45, 41] = 0
    before = len(obj.events)
    assert obj.invoke("PlaceSmallPile", 45, 40, 59, 1, 185) is False
    assert [e["method"] for e in obj.events[before:] if e["event"] == "enter"] == [
        "PlaceSmallPile",
        "SolidTile2",
    ]


def test_valid_pile_framing_and_unknown_destruction_fail_closed():
    obj = fixture()
    assert obj.invoke("PlaceSmallPile", 30, 40, 60, 1, 185)
    before = obj.cells.copy()
    obj.invoke("SquareTileFrame", 30, 40, True)
    assert np.array_equal(before, obj.cells)
    assert any(e["method"] == "Check2x1" for e in obj.events)
    obj.cells["sTileHeader"][30, 41] = 0
    with pytest.raises(UnsupportedCallError, match="Check2x1 destruction"):
        obj.invoke("SquareTileFrame", 30, 40, True)
    obj = fixture()
    assert obj.invoke("PlaceSmallPile", 30, 40, 72, 0, 185)
    obj.invoke("CheckPile", 30, 40)
    obj.cells["sTileHeader"][30, 41] = 0
    with pytest.raises(UnsupportedCallError, match="CheckPile destruction"):
        obj.invoke("CheckPile", 30, 40)


@pytest.mark.parametrize(
    "method,args",
    [
        ("PlaceTile", (30, 40, 186, True, False, -1, 49)),
        ("PlaceTile", (30, 40, 187, True, False, -1, 46)),
        ("PlaceTile", (30, 40, 187, False, False, -1, 49)),
        ("PlaceTile", (30, 40, 187, True, True, -1, 49)),
        ("PlaceTile", (30, 40, 187, True, False, 0, 49)),
        ("PlaceSmallPile", (30, 40, 72, 1, 185)),
        ("PlaceSmallPile", (30, 40, 72, 0, 186)),
        ("Place3x2", (30, 40, 187, 0)),
        ("GrowLivingTree_MakePassage", ()),
    ],
)
def test_unsupported_cases_never_mutate_cells_or_rng(method, args):
    obj = fixture()
    cells, rng, state = obj.cells.copy(), obj.rng.state(), copy.deepcopy(obj.state)
    with pytest.raises(UnsupportedCallError):
        obj.invoke(method, *args)
    assert np.array_equal(cells, obj.cells) and obj.rng.state() == rng and obj.state == state


def test_bounds_and_framing_tables_reject_unimplemented_dependencies():
    obj = fixture()
    before = obj.cells.copy()
    obj.invoke("Place3x2", 4, 40, 187, 47)
    obj.invoke("TileFrame", 5, 40, True, False)
    assert np.array_equal(before, obj.cells)
    assert obj.invoke("SolidTileAllowBottomSlope", -1, 40) is True
    assert obj.invoke("InvalidTileForPilesOrSpeleothems", 1, 40) is False
    with pytest.raises(UnsupportedCallError, match="outside captured region"):
        obj.cell(-1, 0)
    obj.main["tileFrameImportant"][0] = True
    with pytest.raises(UnsupportedCallError, match="TileFrameImportant"):
        obj.invoke("TileFrame", 30, 41, False, False)


def test_entity_collision_uses_integer_pixel_rectangles_and_player_flags():
    obj = fixture()
    entity = {
        "fields": {"active": True, "dead": False, "ghost": False, "width": 18, "height": 40},
        "position": {"X": 480.9, "Y": 640.9},
    }
    obj.state["entities"]["player"].append(entity)
    assert obj.invoke("EmptyTile", 30, 40, False) is False
    entity["fields"]["ghost"] = True
    assert obj.invoke("EmptyTile", 30, 40, False) is True
    obj.state["entities"]["npc"].append(copy.deepcopy(entity))
    assert obj.invoke("EmptyTile", 30, 40, False) is False
    assert obj.invoke("EmptyTile", 29, 40, False) is True  # Touching edge is not overlap.


def test_nested_comparison_keeps_return_rng_and_mutable_native_arguments():
    tile = {name: 0 for name in RAW_DTYPE.names}
    tile["collisionType"] = -1
    rng = UnifiedRandom(19).state()
    events = [
        {
            "event": "enter",
            "id": 12,
            "parent": 9,
            "method": "TileFrameImportant",
            "args": [30, 40, 187, tile, False],
            "rng": rng,
        },
        {
            "event": "exit",
            "id": 12,
            "parent": 9,
            "method": "TileFrameImportant",
            "args": [30, 40, 187, tile, False],
            "result": None,
            "rng": copy.deepcopy(rng),
        },
    ]
    norm = normalized_events(events)
    assert norm[0]["depth"] == norm[1]["depth"] == 0
    assert "collisionType" not in norm[0]["args"][3]
    assert first_difference(norm, norm) is None
    altered = copy.deepcopy(norm)
    altered[1]["rng"]["inext"] += 1
    assert first_difference(norm, altered)["index"] == 1
    assert first_difference(norm, norm[:1])["index"] == 1


def test_living_tree_integration_advances_to_passage_without_mutating_input():
    # Build a complete synthetic pre-pass context, without runtime fixture data.
    from tests.test_living_trees_replay import synthetic

    cells, meta = synthetic(314159265)
    object_state = fixture().state
    for group in ("Terraria.Main", "Terraria.WorldGen", "Terraria.ID.TileID+Sets"):
        for key, value in object_state["globals"][group].items():
            meta["globals"][group].setdefault(key, value)
    meta["globals"]["Terraria.Main"].update(player=[], npc=[])
    meta["chests"] = []
    original = cells.copy()
    replay = LivingTreesReplay(cells, meta)
    with pytest.raises(UnsupportedCallError) as caught:
        replay.grow(120, 199)
    assert caught.value.method == "GrowLivingTree_MakePassage"
    assert replay.object_calls and all("result" in c for c in replay.object_calls)
    assert "result" not in replay.trace[0]
    assert np.array_equal(cells, original)
    assert any(c["result"] for c in replay.object_calls)
