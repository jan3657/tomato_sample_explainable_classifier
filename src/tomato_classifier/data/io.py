"""Dataset I/O and feature/target extraction."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def _detect_csv_delimiter(path: Path) -> str:
    """Infer CSV delimiter from the first line."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        header = sample.splitlines()[0] if sample else ""
        return ";" if header.count(";") > header.count(",") else ","


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".csv":
        delimiter = _detect_csv_delimiter(p)
        csv_kwargs: dict[str, object] = {
            "sep": delimiter,
            "encoding": "utf-8-sig",
        }
        if delimiter == ";":
            # Handle European numeric format, e.g. "1.234,56".
            csv_kwargs["decimal"] = ","
            csv_kwargs["thousands"] = "."
        return pd.read_csv(p, **csv_kwargs)
    raise ValueError(f"Unsupported data format: {suffix}")


def build_xy(df: pd.DataFrame, sample_col: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split frame into feature matrix and target vector."""
    X = df.drop(columns=[sample_col, target_col])
    y = df[target_col]
    return X, y
