"""
StructureMap: exclusion zone system for preventing structure overlap.

Replicates WorldBuilding.StructureMap.AddProtectedStructure(Rectangle, int)
from the game's generation pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class Rectangle:
    """Axis-aligned bounding box for a protected structure."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def intersects(self, other: "Rectangle") -> bool:
        """Check if two rectangles overlap."""
        return (
            self.x < other.right
            and self.right > other.x
            and self.y < other.bottom
            and self.bottom > other.y
        )

    def expandedBy(self, padding: int) -> "Rectangle":
        """Return a new rectangle expanded by padding on all sides."""
        return Rectangle(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )


@dataclass
class StructureMap:
    """Manages protected structure exclusion zones.

    During Terraria worldgen, structures register their bounding rectangles
    via AddProtectedStructure. Subsequent placement attempts check CanPlace
    to avoid overlap. This prevents dungeons from spawning inside temples,
    cabins from overlapping floating islands, etc.
    """

    protectedZones: list[Rectangle] = field(default_factory=list)

    def addProtectedStructure(self, rect: Rectangle, padding: int = 0) -> None:
        """Register a structure's bounding box as protected.

        Args:
            rect: The structure's axis-aligned bounding box.
            padding: Extra buffer tiles around the structure.
        """
        expanded = rect.expandedBy(padding) if padding > 0 else rect
        self.protectedZones.append(expanded)

    def canPlace(self, rect: Rectangle, padding: int = 0) -> bool:
        """Check if a rectangle can be placed without overlapping protected zones.

        Args:
            rect: Proposed placement rectangle.
            padding: Extra buffer to check around the proposed rectangle.

        Returns:
            True if placement is safe (no overlaps).
        """
        testRect = rect.expandedBy(padding) if padding > 0 else rect
        for zone in self.protectedZones:
            if testRect.intersects(zone):
                return False
        return True

    def clear(self) -> None:
        """Reset all protected zones."""
        self.protectedZones.clear()
