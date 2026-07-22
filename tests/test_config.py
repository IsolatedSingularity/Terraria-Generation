from terraforge.config import WorldConfig, WorldScale, seed_to_uint32


def test_seed_conversion_is_stable_and_numeric_strings_match_integers() -> None:
    assert seed_to_uint32("TerraForge") == seed_to_uint32("TerraForge")
    assert seed_to_uint32("42") == seed_to_uint32(42) == 42
    assert seed_to_uint32(-1) == 0xFFFFFFFF


def test_world_scales_have_expected_dimensions() -> None:
    preview = WorldConfig(scale=WorldScale.PREVIEW)
    small = WorldConfig(scale=WorldScale.SMALL)

    assert (preview.width, preview.height) == (240, 140)
    assert (small.width, small.height) == (4200, 1200)
