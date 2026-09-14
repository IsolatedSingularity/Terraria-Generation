"""Lossless import, native state retention and source-derived extraction/render tests."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import fidelity_wld325 as audit_reader
from scripts.render_vanilla_references import CELL_HASH
from terraexplorer.fidelity import wld325
from terraexplorer.fidelity.render import fallback_color, render_crop
from terraexplorer.fidelity.structures import components, extract_structures, statistics
from terraexplorer.fidelity.world import CELL_DTYPE, import_world

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "audit/runtime-oracle-20260914/run-1/bootstrap-small.wld"


@pytest.fixture(scope="module")
def world():
    if not FIXTURE.exists():
        pytest.skip("Local genuine oracle fixture unavailable; no replacement world generated")
    return import_world(FIXTURE)


def test_single_decoder_and_exact_stream(world):
    assert audit_reader.decode_tiles is wld325.decode_tiles
    assert audit_reader.read_world is wld325.read_world
    assert world.cells.shape == (1200, 4200) and world.cells.size == 5040000
    assert world.metadata["version"] == 325
    assert world.metadata["game_mode"] == 0 and not world.metadata["crimson"]
    assert world.semantic_sha256 == CELL_HASH
    assert world.cells.T.copy().tobytes() == world.cell_bytes
    assert world.original_bytes() == FIXTURE.read_bytes()
    assert world.cells.dtype == CELL_DTYPE and CELL_DTYPE.itemsize == 16
    with pytest.raises(ValueError):
        world.cells[0, 0]["tile"] = 1
    with pytest.raises(ValueError):
        world.cells.setflags(write=True)


def test_independent_historical_decoder_field_equivalence(world):
    snapshot = ROOT / "audit/canonical-import-20260914/original_fidelity_wld325.py"
    if not snapshot.exists():
        pytest.skip("Pre-promotion local decoder snapshot unavailable")
    spec = importlib.util.spec_from_file_location("historical_reader", snapshot)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_report, old_cells, *_ = old.read_world(FIXTURE)
    assert bytes(old_cells) == world.cell_bytes
    old_view = np.frombuffer(old_cells, CELL_DTYPE).reshape(4200, 1200).T
    for name in CELL_DTYPE.names:
        assert np.array_equal(old_view[name], world.cells[name]), name
    assert old_report["canonical_cells_sha256"] == world.semantic_sha256


def test_deterministic_fresh_import_and_extraction(world):
    second = import_world(FIXTURE)
    assert second.cell_bytes == world.cell_bytes
    a, rejected = extract_structures(world)
    b, rejected_b = extract_structures(second)
    assert rejected == rejected_b
    assert len(a) == 3
    for first, last in zip(a, b, strict=True):
        assert first.bbox == last.bbox
        assert np.array_equal(first.indices, last.indices)
        left, top, right, bottom = first.bbox
        assert 0 <= left < right <= 4200 and 0 <= top < bottom <= 1200
        # These are whole extracted geometries, not tile-presence assertions.
        assert statistics(world, first) == statistics(second, last)
    assert [s.bbox for s in a] == [
        (500, 108, 891, 846),
        (2593, 462, 2799, 655),
        (549, 140, 628, 379),
    ]
    tree = statistics(world, a[-1])
    assert len(tree["stem_analysis"]["stems"]) == 2
    assert (
        sum(s["associated_support_cells"] for s in tree["stem_analysis"]["stems"])
        == tree["support_cells"]
    )


def test_connected_components_no_row_wrap():
    mask = np.array([[False, True], [True, False]])
    assert len(components(mask)) == 2
    assert len(components(mask, diagonal=True)) == 1


def test_native_unknown_high_ids_and_all_fields_round_trip(world, tmp_path):
    # A synthetic serializer fixture, explicitly not an oracle generation.
    # Native ID 60000 is unknown to our palette; its frame bit is supplied by the file.
    count = 60001
    bits = bytearray((count + 7) // 8)
    bits[60000 // 8] |= 1 << (60000 % 8)
    raw = bytes([0xBF, 0x5F, 0xFF, 0x1E])
    raw += struct.pack("<HhhBBBBBh", 60000, -32768, 32767, 31, 50000 & 255, 30, 0, 50000 >> 8, 1199)
    sections = list(world.section_bytes)
    sections[1] = raw * 4200
    start = 4 + 20 + 2 + 44 + 2 + len(bits)
    pointers = []
    for section in sections:
        pointers.append(start)
        start += len(section)
    preamble = world.preamble[:20] + struct.pack("<h11iH", 11, *pointers, count) + bits
    data = struct.pack("<i", 325) + preamble + b"".join(sections)
    path = tmp_path / "synthetic-native-ids.wld"
    path.write_bytes(data)
    imported = import_world(path)
    assert imported.original_bytes() == data
    expected = (1, 60000, 50000, -32768, 32767, 31, 30, 4, 0, 5, 63, 30)
    assert tuple(imported.cells[0, 0]) == expected
    assert tuple(imported.cells[-1, -1]) == expected
    assert imported.frame_saved.all()
    assert (imported.slope == 4).all() and not imported.half_block.any()
    assert imported.cell_bytes == wld325.CELL.pack(*expected) * 5040000
    image, provenance = render_crop(imported, (0, 0, 3, 3), scale=4, title="Synthetic test")
    assert provenance["unsupported_active_tile_ids"] == [60000]
    assert provenance["unsupported_wall_ids"] == [50000]
    assert image.size[0] >= 12
    assert fallback_color(60000) != fallback_color(60001)


@pytest.mark.parametrize("kind", ["version", "dimensions", "pointers", "importance", "truncated"])
def test_malformed_file_rejected(world, tmp_path, kind):
    data = bytearray(FIXTURE.read_bytes())
    if kind == "version":
        struct.pack_into("<i", data, 0, 324)
    elif kind == "dimensions":
        struct.pack_into("<i", data, 243, 0)  # maxTilesX, after bounds and maxTilesY
    elif kind == "pointers":
        struct.pack_into("<i", data, 30, 1)
    elif kind == "importance":
        struct.pack_into("<H", data, 70, 0)
    else:
        del data[-5:]
    path = tmp_path / f"{kind}.wld"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        import_world(path)


def test_render_uses_imported_bytes_without_generation(world, monkeypatch):
    from terraexplorer.pipeline import TerraExplorerPipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Procedural world generation invoked")

    monkeypatch.setattr(TerraExplorerPipeline, "generate", forbidden)
    bbox = (735, 200, 747, 216)
    a, proof = render_crop(world, bbox, scale=4, title="Provenance test")
    b, proof_b = render_crop(world, bbox, scale=4, title="Provenance test")
    assert a.tobytes() == b.tobytes() and proof == proof_b
    expected = world.cells[200:216, 735:747].T.copy().tobytes()
    assert proof["crop_cell_sha256"] == hashlib.sha256(expected).hexdigest()
    assert proof["source_cells_sha256"] == CELL_HASH
    with pytest.raises(TypeError):
        render_crop(object(), bbox)


def test_regression_statistics_artifact(world):
    path = ROOT / "docs/fidelity/reference_renders/structure_statistics.json"
    if not path.exists():
        pytest.skip("Reference artifact has not yet been exported")
    recorded = json.loads(path.read_text())
    structures, rejected = extract_structures(world)
    # JSON canonicalizes tuples and keeps all histograms, cavity sizes and extents.
    measured = json.loads(json.dumps([statistics(world, s) for s in structures]))
    assert measured == recorded["structures"]
    assert json.loads(json.dumps(rejected)) == recorded["rejected_material_components"]
