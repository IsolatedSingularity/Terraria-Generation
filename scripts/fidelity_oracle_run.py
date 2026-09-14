"""Launch the pinned developer oracle once, preserving a fresh local audit directory."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(output: Path, timeout: float) -> int:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "audit").resolve()):
        raise ValueError("Oracle output must be a fresh directory under workspace audit/")
    lock = json.loads((ROOT / "docs/fidelity/TARGET_LOCK.json").read_text())
    corpus = ROOT / lock["corpus_root_relative"]
    checks = {
        path: sha256(corpus / path) == expected
        for path, expected in {**lock["hashes"], **lock["source_sha256"]}.items()
    }
    checks.update(
        {
            entry["path"]: sha256(ROOT / entry["path"]) == entry["sha256"]
            for entry in lock["artifacts"].values()
        }
    )
    if not all(checks.values()):
        raise ValueError(f"Pinned hash mismatch: {[p for p, ok in checks.items() if not ok]}")
    output.mkdir(parents=True, exist_ok=False)
    preserved = [
        *(ROOT / "docs/fidelity").rglob("*"),
        *(ROOT / "audit/fidelity-bootstrap/runtime-attempt-1").rglob("*"),
    ]
    baseline = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "preserved_sha256": {
            p.relative_to(ROOT).as_posix(): sha256(p) for p in preserved if p.is_file()
        },
        "pin_checks": checks,
    }
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    world = output / "bootstrap-small.wld"
    config = output / "serverconfig.txt"
    config.write_text(
        "\n".join(
            [
                f"world={world}",
                f"worldpath={output}",
                "autocreate=1",
                "seed=1.1.1.0.314159265",
                "difficulty=0",
                "worldname=FidelityBootstrap",
                "maxplayers=1",
                "port=17789",
                "upnp=0",
                "ip=127.0.0.1",
                f"banlist={output / 'banlist.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    server = ROOT / lock["artifacts"]["TerrariaServer.exe"]["path"]
    command = [str(server), "-config", str(config), "-savedirectory", str(output / "saves")]
    record = {
        "started_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "cwd": str(output),
        "launcher_sha256": sha256(Path(__file__)),
        "server_sha256": sha256(server),
        "config_sha256": sha256(config),
        "timeout_seconds": timeout,
        "settings": lock["fixture_settings_intended"],
        "events": [],
    }
    start = time.monotonic()
    previous_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    process = None
    try:
        with (
            (output / "stdout.txt").open("wb") as stdout,
            (output / "stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=output,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            record["pid"] = process.pid
            (output / "launch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            stop_sent = None
            while process.poll() is None:
                elapsed = time.monotonic() - start
                log = (output / "stdout.txt").read_text(errors="replace")
                if stop_sent is None and "Listening on port" in log and world.exists():
                    record["saved_world_sha256_before_shutdown"] = sha256(world)
                    record["generation_and_load_seconds"] = elapsed
                    process.stdin.write(b"exit-nosave\n")
                    process.stdin.flush()
                    stop_sent = elapsed
                    record["events"].append({"seconds": elapsed, "stdin": "exit-nosave\n"})
                    print(
                        f"World saved and loaded after {elapsed:.2f}s; requested exit-nosave",
                        flush=True,
                    )
                if elapsed > timeout or (stop_sent is not None and elapsed - stop_sent > 30):
                    record["events"].append(
                        {"seconds": elapsed, "action": "timeout: kill owned child"}
                    )
                    process.kill()
                    break
                time.sleep(0.25)
            process.wait()
            record["child_exit_code"] = process.returncode
            record["child_exit_hex"] = hex(process.returncode & 0xFFFFFFFF)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        ctypes.windll.kernel32.SetErrorMode(previous_mode)
    record["elapsed_seconds"] = time.monotonic() - start
    record["world_path"] = str(world)
    record["world_exists"] = world.exists()
    record["world_size"] = world.stat().st_size if world.exists() else None
    record["world_sha256"] = sha256(world) if world.exists() else None
    record["stdout_sha256"] = sha256(output / "stdout.txt")
    record["stderr_sha256"] = sha256(output / "stderr.txt")
    record["stderr"] = (output / "stderr.txt").read_text(errors="replace")
    record["shutdown_status"] = (
        "CLEAN_EXIT" if record.get("child_exit_code") == 0 else "FORCED_OR_FAILED_EXIT"
    )
    record["status"] = (
        "SAVED_AND_LOADED"
        if (
            record.get("generation_and_load_seconds")
            and record["world_sha256"] == record.get("saved_world_sha256_before_shutdown")
        )
        else "BLOCKED"
    )
    (output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record["status"] == "SAVED_AND_LOADED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.timeout))
