"""
Market data loader for the FOB USG-NWE Voyage P&L Model.
"""

from __future__ import annotations

import pandas as pd

from .config import MATRIX_COLS


def load_matrix(path: str) -> pd.DataFrame:
    """
    §3.5.1 — Load and validate the historical dataset.

    Reads a CSV or Excel file, sets the 'date' column as a DatetimeIndex
    (ascending), and validates that all columns in config.MATRIX_COLS are
    present. Returns the DataFrame ready for simulate.outer_loop().

    Parameters
    ----------
    path : Absolute or relative path to the market data file.
           .xlsx / .xls  → pd.read_excel()
           all others    → pd.read_csv()

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns in MATRIX_COLS order.
    """
    lower = path.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    missing = [col for col in MATRIX_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Historical dataset is missing required columns: {missing}"
        )

    return df[MATRIX_COLS]
