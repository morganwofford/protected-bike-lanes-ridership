from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class IngestResult:
    df: pd.DataFrame
    files_read: list[str]
    files_failed: list[tuple[str, str]]  # (filepath, error)


def _snake_case(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    # collapse duplicate underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def discover_csvs(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir.resolve()}")
    return sorted([p for p in raw_dir.rglob("*.csv") if p.is_file()])


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Read CSV robustly:
    - try UTF-8, fall back to latin-1
    - don't guess dtypes too aggressively
    """
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def ingest_raw_csvs(
    raw_dir: str | Path,
    *,
    add_source_file: bool = True,
    normalize_columns: bool = True,
) -> IngestResult:
    csv_paths = discover_csvs(raw_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {Path(raw_dir).resolve()}")

    frames: list[pd.DataFrame] = []
    files_read: list[str] = []
    files_failed: list[tuple[str, str]] = []

    for p in csv_paths:
        try:
            df = read_csv_safely(p)

            if normalize_columns:
                df.columns = [_snake_case(c) for c in df.columns]

            if add_source_file:
                df["source_file"] = str(p.as_posix())

            frames.append(df)
            files_read.append(str(p.as_posix()))
        except Exception as e:
            files_failed.append((str(p.as_posix()), repr(e)))

    if not frames:
        raise RuntimeError(f"All CSV reads failed. First error: {files_failed[0] if files_failed else 'unknown'}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return IngestResult(df=combined, files_read=files_read, files_failed=files_failed)
