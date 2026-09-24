"""Small readers and row helpers shared by experiment scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ordered_fieldnames(rows: Sequence[Mapping]) -> list[str]:
    """Return keys in first-seen order across heterogeneous rows."""
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: Mapping, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_torch_load(path: Path, map_location: str = "cpu") -> Any:
    """Load both legacy and current PyTorch checkpoints."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def write_rows_atomic(path: Path, rows: Sequence[Mapping]) -> None:
    """Atomically replace CSV with the same field ordering as write_csv_rows."""
    temporary = path.with_name(f".{path.name}.tmp")
    write_csv_rows(temporary, rows)
    os.replace(temporary, path)
