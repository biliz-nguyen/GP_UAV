"""Small deterministic IO helpers shared by the pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Represent unknown/nonfinite values as JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, bytes):
        return value.hex()
    return value


def write_json(path: str | Path, data) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(json_safe(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else json_safe(value)
                for key, value in row.items()
            })


def git_revision(project_dir: str | Path | None = None) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, capture_output=True,
            check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
