from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Container for the result of a raw CSV ingest run.
# This keeps the combined dataframe along with metadata about which
# files succeeded or failed, so we can audit data quality later.
@dataclass(frozen=True)
class IngestResult:
    df: pd.DataFrame
    files_read: list[str]
    files_failed: list[tuple[str, str]]  # (filepath, error msg)


def _snake_case(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    # collapse duplicate underscores in names
    while "__" in s:
        s = s.replace("__", "_")
    return s


def discover_csvs(raw_dir: str | Path) -> list[Path]:
    # Recursively find all CSV files under the raw data directory.
    # This allows the raw data folder to contain subfolders by year,
    # counter location, etc.
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir.resolve()}")
    return sorted([p for p in raw_dir.rglob("*.csv") if p.is_file()])


def read_csv_safely(path: Path) -> pd.DataFrame:
    # Read a CSV with basic robustness:
    # - Try UTF-8 first (most files)
    # - Fall back to latin-1 for legacy or badly-encoded city data
    # - Let pandas infer dtypes later during cleaning, not at ingest time
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
     """
    Load all raw CSV files under `raw_dir` into a single dataframe.

    This function does the following:
    - Discovers every CSV recursively
    - Reads each file with tolerant encoding handling
    - Normalizes column names to snake_case
    - Optionally adds a `source_file` column for traceability
    - Tracks which files failed to load and why

    Returns:
        IngestResult with:
          - df: combined dataframe of all CSVs
          - files_read: list of successfully loaded file paths
          - files_failed: list of (path, error) tuples for failures
    """
    csv_paths = discover_csvs(raw_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {Path(raw_dir).resolve()}")

    frames: list[pd.DataFrame] = []
    files_read: list[str] = []
    files_failed: list[tuple[str, str]] = []

    for p in csv_paths:
        try:
            df = read_csv_safely(p)
            # Standardize column names so we can concatenate files
            # even if their schemas vary slightly
            if normalize_columns:
                df.columns = [_snake_case(c) for c in df.columns]
            # Preserve provenance: every row knows which file it came from
            if add_source_file:
                df["source_file"] = str(p.as_posix())

            frames.append(df)
            files_read.append(str(p.as_posix()))
        except Exception as e:
            # Never fail silently. Record which files broke and why.
            files_failed.append((str(p.as_posix()), repr(e)))

    if not frames:
        raise RuntimeError(f"All CSV reads failed. First error: {files_failed[0] if files_failed else 'unknown'}")
    
    # Concatenate all CSVs into a single wide table.
    # sort=False preserves original column ordering across files.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return IngestResult(df=combined, files_read=files_read, files_failed=files_failed)
