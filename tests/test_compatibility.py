from terraexplorer import TerraExplorerPipeline
from terraforge import TerraForgePipeline
from terraforge.config import WorldConfig


def test_terraforge_aliases_resolve_to_terraexplorer() -> None:
    assert TerraForgePipeline is TerraExplorerPipeline
    world = TerraForgePipeline().generate(WorldConfig(seed="compatibility"))

    assert world.metadata["seed"] == "compatibility"
