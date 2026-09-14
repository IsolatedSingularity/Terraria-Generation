"""Independent ordinary LivingTrees replay prefix for pinned Terraria 1.4.5.7.

This is intentionally incomplete: unknown object cases and connected passages
stop with an explicit dependency boundary. It is never used by the approximate
generator. The caller owns a mutable copy of a captured native Tile array.
"""

from __future__ import annotations

import math

import numpy as np

from terraexplorer.fidelity.pass_snapshot import RAW_DTYPE
from terraexplorer.fidelity.unified_random import UnifiedRandom


class UnsupportedCallError(RuntimeError):
    def __init__(self, method, args, rng):
        self.method, self.arguments, self.rng = method, list(args), rng.state()
        super().__init__(f"Independent implementation stops before {method}{tuple(args)}")


class LivingTreesReplay:
    """Small ordinary-world compatibility; fail closed on secret-seed inputs."""

    def __init__(self, cells, metadata):
        if cells.dtype != RAW_DTYPE or cells.ndim != 2:
            raise ValueError("Expected native Tile [x,y] array")
        self.cells = cells.copy()
        self.width, self.height = cells.shape
        if (metadata["width"], metadata["height"]) != cells.shape:
            raise ValueError("Snapshot dimensions disagree")
        self.rng = UnifiedRandom.from_state(metadata["rng"])
        globals_ = metadata["globals"]
        main = globals_["Terraria.Main"]
        world = globals_["Terraria.WorldGen"]
        gen = globals_["Terraria.WorldBuilding.GenVars"]
        sets = globals_["Terraria.ID.TileID+Sets"]
        for name in (
            "drunkWorldGen",
            "notTheBees",
            "tenthAnniversaryWorldGen",
            "remixWorldGen",
            "skyblockWorldGen",
        ):
            if world[name]:
                raise ValueError(f"Unsupported secret world flag: {name}")
        data = metadata["world_file_data"]
        secret_seeds = metadata["secret_seeds"]
        if not secret_seeds or any(secret_seeds.values()):
            raise ValueError("Explicit disabled secret-seed options required")
        for name in (
            "DrunkWorld",
            "NotTheBees",
            "ForTheWorthy",
            "Anniversary",
            "DontStarve",
            "RemixWorld",
            "NoTrapsWorld",
            "ZenithWorld",
            "SkyblockWorld",
        ):
            if data[name]:
                raise ValueError(f"Unsupported secret world flag: {name}")
        if data["GameMode"] != 0 or not data["HasCorruption"] or data["IsHardMode"]:
            raise ValueError("Only Classic Corruption pre-Hardmode replay is supported")
        if data.get("_seedText") != str(data.get("_seed")):
            raise ValueError("Ordinary numeric seed metadata required")
        self.surface = main["worldSurface"]
        self.beach = world["beachDistance"]
        self.caves = gen["mCaveX"][: gen["numMCaves"]]
        self.solid_types = list(main["tileSolid"])
        self.solid_top = main["tileSolidTop"]
        self.dungeon_walls = main["wallDungeon"]
        self.ores = sets["Ore"]
        self.clouds = sets["Clouds"]
        self.trace = []
        self.candidates = []
        self.metadata = metadata
        self.placement = None
        self.object_calls = []

    def active(self, x, y):
        return bool(int(self.cells["sTileHeader"][x, y]) & 32)

    def solid(self, x, y):
        if not 0 <= x < self.width or not 0 <= y < self.height:
            return False
        cell = self.cells[x, y]
        header, kind = int(cell["sTileHeader"]), int(cell["type"])
        return bool(
            header & 32
            and self.solid_types[kind]
            and not self.solid_top[kind]
            and not header & (1024 | 28672 | 64)
        )

    def place_wood(self, x, y, protect=True, kind=191):
        if protect and self.dungeon_walls[int(self.cells["wall"][x, y])]:
            return
        self.cells["type"][x, y] = kind
        self.cells["sTileHeader"][x, y] = (int(self.cells["sTileHeader"][x, y]) | 32) & ~1024

    def can_place_leaves(self, x, y):
        if not (5 <= x < self.width - 5 and 5 <= y < self.height - 5):
            return False
        cell = self.cells[x, y]
        wall, kind = int(cell["wall"]), int(cell["type"])
        if wall in (244, 78) or self.dungeon_walls[wall]:
            return False
        return not self.active(x, y) or (kind != 191 and not self.clouds[kind])

    def unsupported(self, method, *args):
        """Resolve only validated object cases; retain every other boundary."""
        if method in ("PlaceTile", "PlaceSmallPile"):
            if self.placement is None:
                from terraexplorer.fidelity.object_placement import ObjectPlacement

                main = self.metadata["globals"]["Terraria.Main"]
                entities = {
                    name: [
                        {"fields": entity, "position": entity["position"]}
                        for entity in main[name]
                        if entity["active"]
                    ]
                    for name in ("player", "npc")
                }
                self.placement = ObjectPlacement(
                    self.cells,
                    self.rng,
                    {
                        "globals": self.metadata["globals"],
                        "entities": entities,
                        "chests": self.metadata["chests"],
                    },
                )
                self.placement.main["tileSolid"] = self.solid_types
            call = {"method": method, "args": list(args), "rng_before": self.rng.state()}
            self.object_calls.append(call)
            result = self.placement.invoke(method, *args)
            call.update(result=result, rng_after=self.rng.state())
            return result
        raise UnsupportedCallError(method, args, self.rng)

    def grow(self, x, y, patch=False):
        before = {"method": "GrowLivingTree", "args": [x, y, patch], "rng_before": self.rng.state()}
        self.trace.append(before)
        result = self._grow(x, y, patch)
        before.update(result=result, rng_after=self.rng.state())
        return result

    def _grow(self, x, y, patch):
        r = self.rng.next
        if not self.solid(x, y + 1) or self.active(x, y):
            return False
        support = int(self.cells["type"][x, y + 1])
        if support not in (0, 2, 1, 40) and not self.ores[support]:
            return False
        if y < 150:
            return False
        left, right = x - r(2, 3), x + r(2, 3)
        if r(5) == 0:
            if r(2) == 0:
                left -= 1
            else:
                right += 1
        trunk_width = right - left
        passage = trunk_width >= 4
        clearance = 20 if patch else 50
        if patch:
            left, right = x - r(1, 3), x + r(1, 3)
        for i in range(x - clearance, x + clearance + 1):
            for j in range(5, y - 5):
                if self.active(i, j) and (
                    not patch or int(self.cells["type"][i, j]) not in (2, 0, 1, 191, 192, 383, 384)
                ):
                    return False
        self.solid_types[48] = False
        inner_left, inner_right = left, right
        root_left, root_right = left, right
        row = y
        growing = True
        branch_wait, side, interval = r(-8, -4), r(2), r(5, 15)
        branches = []
        leaves = []
        while growing:
            branch_wait += 1
            if branch_wait > interval:
                interval, branch_wait = r(5, 15), 0
                branch_y = row + r(5)
                if r(5) == 0:
                    side = 1 - side
                if side == 0:
                    branches.append((left, branch_y, -1, right - left))
                    if r(2) == 0:
                        left += 1
                    inner_left += 1
                    side = 1
                else:
                    branches.append((right, branch_y, 1, right - left))
                    if r(2) == 0:
                        right -= 1
                    inner_right -= 1
                    side = 0
                if inner_left == inner_right:
                    growing = False
            for i in range(left, right + 1):
                self.place_wood(i, row)
            row -= 1
        for bx, by, direction, width in branches[:-1]:
            bx += direction
            remaining = int(width * (1.0 + r(20, 30) * 0.1))
            self.place_wood(bx, by + 1)
            bud_wait = r(3, 5)
            while remaining > 0:
                remaining -= 1
                self.place_wood(bx, by)
                if r(10) == 0:
                    by += 1 if r(2) != 0 else -1
                else:
                    bx += direction
                if bud_wait > 0:
                    bud_wait -= 1
                elif r(2) == 0:
                    bud_wait = r(2, 5)
                    if not self.dungeon_walls[int(self.cells["wall"][bx, by])]:
                        bud_dy = -1 if r(2) == 0 else 1
                        self.place_wood(bx, by, False)
                        self.place_wood(bx, by + bud_dy, False)
                        leaves.append((bx, by, False))
                if remaining == 0:
                    leaves.append((bx, by, False))
        top_x, top_y = (left + right) // 2, row
        remaining = r(trunk_width * 3, trunk_width * 5)
        left_wait = right_wait = 0
        while remaining > 0 and top_y >= 30:
            self.place_wood(top_x, top_y)
            left_wait, right_wait = max(0, left_wait - 1), max(0, right_wait - 1)
            for direction in (-1, 1):
                if (left_wait if direction < 0 else right_wait) != 0 or r(2) != 0:
                    continue
                bx, by = top_x, top_y
                length = r(trunk_width, trunk_width * 3)
                if direction < 0:
                    left_wait = r(3, 5)
                else:
                    right_wait = r(3, 5)
                bud_wait = 0
                while length > 0:
                    length -= 1
                    bx += direction
                    self.place_wood(bx, by)
                    if length == 0:
                        leaves.append((bx, by, True))
                    if r(5) == 0:
                        by += 1 if r(2) != 0 else -1
                        self.place_wood(bx, by)
                    if bud_wait > 0:
                        bud_wait -= 1
                    elif r(3) == 0:
                        bud_wait = r(2, 4)
                        bud_y = by + (1 if r(2) != 0 else -1)
                        self.place_wood(bx, bud_y)
                        leaves.append((bx, bud_y, True))
                        leaves.append((bx + r(-5, 6), bud_y + r(-5, 6), True))
            leaves.append((top_x, top_y, False))
            if r(4) == 0:
                top_x += 1 if r(2) != 0 else -1
                self.place_wood(top_x, top_y)
            top_y -= 1
            remaining -= 1
        for root_x in range(root_left, root_right + 1):
            depth, root_y = r(1, 6), y + 1
            while depth > 0:
                if self.solid(root_x, root_y):
                    depth -= 1
                self.place_wood(root_x, root_y, False)
                root_y += 1
            base_y = root_y
            count = r(2, trunk_width + 1)
            for _ in range(count):
                root_y = base_y
                center = (root_left + root_right) // 2
                dx = 1 if root_x >= center else -1
                dy = 1
                if root_x == center or (trunk_width > 6 and root_x in (center - 1, center + 1)):
                    dx = 0
                direction, walk_x = dx, root_x
                length = r(int(trunk_width * 3.5), trunk_width * 6)
                while length > 0:
                    length -= 1
                    walk_x += dx
                    if int(self.cells["wall"][walk_x, root_y]) != 244:
                        self.place_wood(walk_x, root_y, False)
                    root_y += dy
                    if int(self.cells["wall"][walk_x, root_y]) != 244:
                        self.place_wood(walk_x, root_y, False)
                    if not self.active(walk_x, root_y + 1):
                        dx, dy = 0, 1
                    if r(3) == 0:
                        if direction < 0:
                            dx = -1 if dx == 0 else 0
                        elif direction > 0:
                            dx = 1 if dx == 0 else 0
                        else:
                            dx = r(-1, 2)
                    if r(3) == 0:
                        dy = 1 if dy == 0 else 0
        for leaf_x, leaf_y, round_shape in leaves:
            radius = int(r(5, 8) * (1.0 + trunk_width * 0.05))
            if round_shape:
                radius = r(6, 12) + trunk_width
            min_x, max_x = leaf_x - radius * 2, leaf_x + radius * 2
            min_y, max_y = leaf_y - radius * 2, leaf_y + radius * 2
            vertical_scale = 2.0 - r(5) * 0.1
            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    if not self.can_place_leaves(i, j):
                        continue
                    dx, dy = leaf_x - i, leaf_y - j
                    inside = (
                        math.sqrt(dx * dx + dy * dy) < radius * 0.9
                        if round_shape
                        else abs(dx) + abs(dy) * vertical_scale < radius
                    )
                    if inside:
                        self.place_wood(i, j, False, 192)
                if r(30) == 0:
                    j = min_y
                    if (
                        5 <= i < self.width - 5
                        and 5 <= j < self.height - 5
                        and not self.active(i, j)
                    ):
                        while not self.active(i, j + 1) and j < max_y:
                            j += 1
                        if int(self.cells["type"][i, j + 1]) == 192:
                            self.unsupported("PlaceTile", i, j, 187, True, False, -1, r(50, 52))
                if round_shape or r(15) != 0:
                    continue
                j = max_y
                limit = j + 100
                if self.active(i, j):
                    continue
                while not self.active(i, j + 1) and j < limit:
                    j += 1
                if int(self.cells["type"][i, j + 1]) == 192:
                    continue
                if r(2) == 0:
                    self.unsupported("PlaceTile", i, j, 187, True, False, -1, r(47, 50))
                    continue
                size = r(2)
                style = r(59, 62) if size == 1 else 72
                self.unsupported("PlaceSmallPile", i, j, style, size, 185)
        if passage:
            opening = any(
                int(self.cells["wall"][i, j]) == 0 and not self.solid(i, j)
                for j in range(y, min(y + 20, math.ceil(self.surface - 2.0)))
                for i in range(root_left, root_right + 1)
            )
            if not opening:
                self.unsupported(
                    "GrowLivingTree_MakePassage", y, trunk_width, root_left, root_right, patch
                )
        self.solid_types[48] = True
        return True

    def _blocked_region(self, x, y):
        region = self.cells[x - 50 : x + 50, y - 50 : y + 50]
        return bool(
            np.any(
                ((region["sTileHeader"] & 32) != 0)
                & np.isin(region["type"], (41, 43, 44, 481, 482, 483, 189, 196, 460, 717, 718, 719))
            )
        )

    def run(self):
        r = self.rng.next
        count = r(0, int(2.0 * (self.width / 4200.0)) + 1)
        if count == 0 and r(2) == 0:
            count += 1
        for _ in range(count):
            accepted, attempts = False, 0
            while not accepted:
                attempts += 1
                if attempts > self.width // 2:
                    accepted = True
                x = r(self.beach, self.width - self.beach)
                self.candidates.append(x)
                if self.width // 2 - 200 < x < self.width // 2 + 200:
                    continue
                y = 0
                while not self.active(x, y) and y < self.surface:
                    y += 1
                if y >= self.surface or int(self.cells["type"][x, y]) != 0:
                    continue
                y -= 1
                if y <= 150:
                    continue
                nearby = self.cells[
                    max(0, x - 10) : min(self.width, x + 11),
                    max(0, y - 10) : min(self.height, y + 11),
                ]
                if np.any(
                    ((nearby["sTileHeader"] & 32) != 0) & np.isin(nearby["type"], (191, 192))
                ):
                    continue
                if self._blocked_region(x, y) or any(c - 50 < x < c + 50 for c in self.caves):
                    continue
                accepted = self.grow(x, y)
                if not accepted:
                    continue
                for direction in (-1, 1):
                    patch_x = x
                    for _ in range(r(4)):
                        patch_x += r(13, 31) * direction
                        if self.width // 2 - 200 < patch_x < self.width // 2 + 200:
                            continue
                        patch_y = y
                        if self.active(patch_x, patch_y):
                            while patch_y > 0 and self.active(patch_x, patch_y):
                                patch_y -= 1
                        else:
                            while patch_y < self.height - 1 and not self.active(patch_x, patch_y):
                                patch_y += 1
                            patch_y -= 1
                        # Vanilla checks the main origin again, not the patch origin.
                        if not self._blocked_region(x, y):
                            self.grow(patch_x, patch_y, True)
