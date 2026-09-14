"""Observed LivingTrees object cases, not the general Terraria placement API.

Native [x,y] storage and absolute world coordinates; local regions reject reads
outside their captured bounds. Source target: Terraria 1.4.5.7 WorldGen/Tile.
"""

from __future__ import annotations

import copy

from terraexplorer.fidelity.living_trees import UnsupportedCallError


class ObjectPlacement:
    def __init__(self, cells, rng, state, origin=(0, 0), world_shape=None):
        self.cells, self.rng = cells, rng
        self.state = copy.deepcopy(state)
        self.origin = tuple(origin)
        self.world_shape = tuple(world_shape or cells.shape)
        self.main = self.state["globals"]["Terraria.Main"]
        self.world = self.state["globals"]["Terraria.WorldGen"]
        self.sets = self.state["globals"]["Terraria.ID.TileID+Sets"]
        self.events = []
        self.depth = 0
        if not self.world["generatingWorld"] or not self.main["dedServ"]:
            self.fail("generation-only object context")

    def fail(self, name, *args):
        raise UnsupportedCallError(name, args, self.rng)

    def cell(self, x, y):
        i, j = x - self.origin[0], y - self.origin[1]
        if not (0 <= i < self.cells.shape[0] and 0 <= j < self.cells.shape[1]):
            self.fail("object read outside captured region", x, y)
        return self.cells[i, j]

    def tile_arg(self, x, y):
        return {name: int(self.cell(x, y)[name]) for name in self.cells.dtype.names}

    def active(self, x, y):
        return bool(int(self.cell(x, y)["sTileHeader"]) & 32)

    def invoke(self, name, *args):
        handlers = {
            "PlaceTile": self.place_tile,
            "PlaceSmallPile": self.small_pile,
            "Place3x2": self.place3x2,
            "SquareTileFrame": self.square_frame,
            "TileFrame": self.tile_frame,
            "TileFrameImportant": self.important,
            "Check3x2": self.check3x2,
            "CheckPile": self.check_pile,
            "Check2x1": self.check2x1,
            "SolidTile2": self.solid2,
            "SolidTileAllowBottomSlope": self.support,
            "InvalidTileForPilesOrSpeleothems": self.invalid_support,
            "EmptyTile": self.empty_tile,
            "Add": self.map_add,
            "KillTile": self.kill_tile,
            "CheckTileBreakability": self.breakability,
            "KillTile_GetTileDustAmount": self.dust_amount,
            "KillTile_MakeTileDust": self.make_dust,
            "AttemptFossilShattering": self.fossil,
            "CheckTileBreakability2_ShouldTileSurvive": self.survive,
            "CheckExploitDestroyQueue": self.exploit_queue,
        }
        if name not in handlers:
            self.fail(name, *args)
        depth = self.depth
        self.events.append(
            {
                "event": "enter",
                "method": name,
                "args": copy.deepcopy(list(args)),
                "depth": depth,
                "rng": self.rng.state(),
            }
        )
        self.depth += 1
        result = handlers[name](*args)
        self.depth -= 1
        if name == "TileFrameImportant":
            args = (*args[:3], self.tile_arg(*args[:2]), args[4])
        self.events.append(
            {
                "event": "exit",
                "method": name,
                "args": copy.deepcopy(list(args)),
                "depth": depth,
                "result": result,
                "rng": self.rng.state(),
            }
        )
        return result

    def solid2(self, x, y):
        t = self.cell(x, y)
        sh, kind = int(t["sTileHeader"]), int(t["type"])
        slope = (sh >> 12) & 7
        return bool(
            sh & 32
            and self.main["tileSolid"][kind]
            and not sh & (64 | 1024)
            and (slope == 0 or (self.sets["Platforms"][kind] and slope in (1, 2)))
        )

    def support(self, x, y):
        if not (0 <= x < self.world_shape[0] and 0 <= y < self.world_shape[1]):
            return True
        t = self.cell(x, y)
        sh, kind = int(t["sTileHeader"]), int(t["type"])
        slope = (sh >> 12) & 7
        if self.sets["Platforms"][kind] and slope in (1, 2):
            self.fail("PlatformProperTopFrame", int(t["frameX"]))
        return bool(
            sh & 32
            and (self.main["tileSolid"][kind] or self.main["tileSolidTop"][kind])
            and slope not in (1, 2)
            and not sh & (64 | 1024)
        )

    def invalid_support(self, x, y):
        if not (2 <= x < self.world_shape[0] - 2 and 2 <= y < self.world_shape[1] - 2):
            return False
        return self.active(x, y) and bool(self.sets["Boulders"][int(self.cell(x, y)["type"])])

    def empty_tile(self, x, y, ignore):
        if self.active(x, y) and not ignore:
            return False
        for category, entities in self.state["entities"].items():
            for entity in entities:
                f, p = entity["fields"], entity["position"]
                if not f["active"] or category == "player" and (f["dead"] or f["ghost"]):
                    continue
                left, top = int(p["X"]), int(p["Y"])
                if (
                    x * 16 < left + f["width"]
                    and left < (x + 1) * 16
                    and y * 16 < top + f["height"]
                    and top < (y + 1) * 16
                ):
                    return False
        return True

    def map_add(self, x, y):
        # MapUpdateQueue.Add returns immediately during dedicated generation.
        if not (self.main["dedServ"] or self.world["generatingWorld"]):
            self.fail("MapUpdateQueue.Add", x, y)

    def clear_block(self, x, y, clear_tile=False):
        t = self.cell(x, y)
        t["sTileHeader"] = int(t["sTileHeader"]) & ~(31 | 1024 | 28672)
        t["bTileHeader3"] = int(t["bTileHeader3"]) & ~160
        if clear_tile:
            t["type"], t["frameX"], t["frameY"] = 0, 0, 0
            t["sTileHeader"] = int(t["sTileHeader"]) & ~32

    def set_object(self, x, y, kind, fx, fy):
        t = self.cell(x, y)
        t["sTileHeader"] = int(t["sTileHeader"]) | 32
        t["type"], t["frameX"], t["frameY"] = kind, fx, fy

    def place_tile(self, x, y, kind, mute, forced, player, style):
        if (kind, mute, forced, player) != (187, True, False, -1) or style not in range(47, 52):
            self.fail("PlaceTile unsupported case", x, y, kind, mute, forced, player, style)
        if self.main["tileSolid"][187] or self.sets["TruncatesWalls"][187]:
            self.fail("PlaceTile unexpected type-187 tables")
        t = self.cell(x, y)
        if self.active(x, y) and int(t["type"]) == 488:
            return False
        self.invoke("EmptyTile", x, y, False)
        if not self.active(x, y):
            self.clear_block(x, y, True)
        elif self.sets["ResetsHalfBrickPlacementAttempt"][187]:
            self.fail("PlaceTile reset-half-brick table")
        self.invoke("Place3x2", x, y, 187, style)
        self.invoke("SquareTileFrame", x, y, True)
        if self.active(x, y):
            if self.sets["TruncatesWalls"][int(t["type"])]:
                self.fail("SquareWallFrame", x, y)
            self.invoke("SquareTileFrame", x, y, True)
            return True
        return False

    def place3x2(self, x, y, kind, style):
        if kind != 187 or style not in range(47, 52):
            self.fail("Place3x2", x, y, kind, style)
        if x < 5 or x > self.world_shape[0] - 5 or y < 5 or y > self.world_shape[1] - 5:
            return
        valid = True
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 1):
                if self.active(i, j):
                    valid = False
            if self.invoke("InvalidTileForPilesOrSpeleothems", i, y + 1):
                valid = False
            if not self.invoke("SolidTile2", i, y + 1):
                valid = False
        if valid:
            for j in range(y - 1, y + 1):
                for i in range(x - 1, x + 2):
                    self.set_object(i, j, 187, 54 * style + 18 * (i - x + 1), 18 * (j - y + 1))

    def small_pile(self, x, y, style, size, kind):
        if kind != 185 or (size, style) not in ((0, 72), (1, 59), (1, 60), (1, 61)):
            self.fail("PlaceSmallPile unsupported case", x, y, style, size, kind)
        t = self.cell(x, y)
        if t["liquid"] > 0 and (int(t["bTileHeader"]) & 96) == 32:
            return False
        valid = self.invoke("SolidTile2", x, y + 1)
        if size == 1:
            valid = (
                valid
                and self.invoke("SolidTile2", x + 1, y + 1)
                and not self.active(x, y)
                and not self.active(x + 1, y)
            )
            if valid and (
                self.invoke("InvalidTileForPilesOrSpeleothems", x, y + 1)
                or self.invoke("InvalidTileForPilesOrSpeleothems", x + 1, y + 1)
            ):
                valid = False
        else:
            valid = valid and not self.active(x, y)
        if valid:
            for i in range(size + 1):
                self.set_object(x + i, y, 185, style * 18 * (size + 1) + 18 * i, 18 * size)
        return bool(valid)

    def square_frame(self, x, y, reset):
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                self.invoke("TileFrame", i, j, reset if (i, j) == (x, y) else False, False)

    def tile_frame(self, x, y, reset, no_break):
        if x <= 5 or y <= 5 or x >= self.world_shape[0] - 5 or y >= self.world_shape[1] - 5:
            return
        self.invoke("Add", x, y)
        if not self.active(x, y):
            self.clear_block(x, y)
            return
        kind = int(self.cell(x, y)["type"])
        if self.main["tileFrameImportant"][kind]:
            if no_break:
                self.fail("TileFrame noBreak", x, y, reset, no_break)
            self.invoke("TileFrameImportant", x, y, kind, self.tile_arg(x, y), reset)
            return
        if not self.main["tileSolid"][kind] and (kind in (49, 80) or self.sets["IsVine"][kind]):
            self.fail("TileFrame non-solid special type", x, y, kind)
        # TileFrameCosmetic is guarded by !generatingWorld in the pinned source.

    def important(self, x, y, kind, tile, reset):
        if kind == 187:
            self.invoke("Check3x2", x, y, kind)
        elif kind == 185:
            self.invoke("CheckPile", x, y)
        else:
            self.fail("TileFrameImportant", x, y, kind, reset)

    def check3x2(self, x, y, kind):
        if self.world["destroyObject"]:
            return
        t = self.cell(x, y)
        style = int(t["frameX"]) // 54
        if kind != 187 or style not in range(47, 52) or t["frameY"] not in (0, 18):
            self.fail("Check3x2 unsupported style", x, y, kind)
        left, top = x - (int(t["frameX"]) // 18) % 3, y - int(t["frameY"]) // 18
        valid = True
        for i in range(left, left + 3):
            for j in range(top, top + 2):
                c = self.cell(i, j)
                if (
                    not self.active(i, j)
                    or c["type"] != kind
                    or c["frameX"] != (i - left) * 18 + style * 54
                    or c["frameY"] != (j - top) * 18
                ):
                    valid = False
            if not self.invoke("SolidTileAllowBottomSlope", i, top + 2):
                valid = False
                continue
            if self.invoke("InvalidTileForPilesOrSpeleothems", i, top + 2):
                valid = False
        if valid:
            return
        self.world["destroyObject"] = True
        for i in range(left, left + 3):
            for j in range(top, top + 2):
                if self.cell(i, j)["type"] == kind and self.active(i, j):
                    self.invoke("KillTile", i, j, False, False, False)
        self.world["destroyObject"] = False
        for i in range(left - 1, left + 4):
            for j in range(top - 1, top + 4):
                self.invoke("TileFrame", i, j, False, False)

    def check_pile(self, x, y):
        t = self.cell(x, y)
        if t["frameY"] == 18:
            self.invoke("Check2x1", x, y, int(t["type"]))
        elif not self.invoke("SolidTileAllowBottomSlope", x, y + 1):
            self.fail("CheckPile destruction", x, y)
        elif int(t["frameX"]) // 18 != 72:
            self.fail("CheckPile unsupported style", x, y)

    def check2x1(self, x, y, kind):
        if self.world["destroyObject"]:
            return
        t = self.cell(x, y)
        left = x - (int(t["frameX"]) // 18) % 2
        a, b = self.cell(left, y), self.cell(left + 1, y)
        if kind != 185 or int(a["frameX"]) // 36 not in (59, 60, 61) or a["frameY"] != 18:
            self.fail("Check2x1 unsupported case", x, y, kind)
        valid = (
            b["frameX"] == a["frameX"] + 18
            and a["type"] == kind
            and b["type"] == kind
            and self.active(left, y)
            and self.active(left + 1, y)
        )
        if not self.invoke("SolidTileAllowBottomSlope", left, y + 1):
            valid = False
        if not self.invoke("SolidTileAllowBottomSlope", left + 1, y + 1):
            valid = False
        for i in (left, left + 1):
            if self.invoke("InvalidTileForPilesOrSpeleothems", i, y + 1):
                valid = False
                break
        if not valid:
            self.fail("Check2x1 destruction", x, y, kind)

    def kill_tile(self, x, y, fail, effect_only, no_item):
        t = self.cell(x, y)
        if not self.active(x, y):
            return
        if t["type"] != 187 or any((fail, effect_only, no_item)) or not self.world["destroyObject"]:
            self.fail("KillTile unsupported case", x, y, fail, effect_only, no_item)
        self.invoke("CheckTileBreakability", x, y)
        amount = self.invoke("KillTile_GetTileDustAmount", False, self.tile_arg(x, y))
        for _ in range(amount):
            self.invoke("KillTile_MakeTileDust", x, y, self.tile_arg(x, y))
        self.invoke("AttemptFossilShattering", x, y, self.tile_arg(x, y), False)
        self.invoke("CheckTileBreakability2_ShouldTileSurvive", x, y)
        # Dedicated generation suppresses drops and dust allocation. Type 187
        # is neither a container, fossil, wiring tile nor wall-truncating tile.
        t["sTileHeader"] = int(t["sTileHeader"]) & ~(32 | 64 | 1024 | 31)
        t["bTileHeader3"] = int(t["bTileHeader3"]) & ~160
        t["bTileHeader2"] = int(t["bTileHeader2"]) & ~48
        t["frameX"], t["frameY"], t["type"] = -1, -1, 0
        self.invoke("SquareTileFrame", x, y, True)
        self.invoke("CheckExploitDestroyQueue")

    def breakability(self, x, y):
        if (
            self.cell(x, y)["type"] != 187
            or self.main["tileSolid"][187]
            or self.main["tileSolidTop"][187]
            or self.active(x, y + 1)
            and self.cell(x, y + 1)["type"] == 10
        ):
            self.fail("CheckTileBreakability unsupported support", x, y)
        return 0  # Non-solid, non-platform branch, after the locked-door guard.

    def dust_amount(self, fail, tile):
        if tile["type"] != 187:
            self.fail("KillTile_GetTileDustAmount", fail, tile)
        return 3 if fail else 10

    def make_dust(self, x, y, tile):
        if tile["type"] != 187 or tile["frameX"] // 54 not in range(47, 52):
            self.fail("KillTile_MakeTileDust", x, y, tile)
        # These styles select default dust type 0 with no random draw. The
        # Dust.NewDust generation/menu guard returns its out-of-pool sentinel.
        if not (self.main["gameMenu"] or self.world["generatingWorld"]):
            self.fail("Dust.NewDust allocation", x, y)
        return 6000

    def fossil(self, x, y, tile, fail):
        if tile["type"] == 404:
            self.fail("AttemptFossilShattering", x, y, tile, fail)

    def survive(self, x, y):
        if self.cell(x, y)["type"] != 187:
            self.fail("CheckTileBreakability2_ShouldTileSurvive", x, y)
        return False

    def exploit_queue(self):
        if not self.world["destroyObject"]:
            self.fail("CheckExploitDestroyQueue outside object destruction")
