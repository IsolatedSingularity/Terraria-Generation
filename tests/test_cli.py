from pathlib import Path

from terraexplorer.cli import main


def test_passes_command_reports_full_catalogue(capsys) -> None:
    assert main(["passes"]) == 0

    output = capsys.readouterr().out
    assert "001  Reset" in output
    assert "107  Final Cleanup" in output


def test_generate_command_writes_requested_outputs(tmp_path: Path) -> None:
    png = tmp_path / "world.png"
    npz = tmp_path / "world.npz"
    metadata = tmp_path / "world.json"

    result = main(
        [
            "generate",
            "--seed",
            "cli-test",
            "--png",
            str(png),
            "--npz",
            str(npz),
            "--json",
            str(metadata),
            "--quiet",
        ]
    )

    assert result == 0
    assert png.is_file()
    assert npz.is_file()
    assert metadata.is_file()
