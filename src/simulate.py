"""
Simulation engine for the FOB USG-NWE Voyage P&L Model.

Inner loop  — single P&L sample via matrix tracing   (§3.5.2–3.5.6)
Outer loop  — N iterations producing the P&L distribution

Matrix reading helpers implement the missing-value rules from §3.5.5:

  Financial single-input  (SOFR, FX, WS Quote, TD25 Flat Rate)
      → nearest previous non-NaN assessment

  Financial 5-day window  (WTI Houston FOB, Dated Brent)
      → anchor at nearest non-NaN at-or-before target; ±2 trading days

  Vessel tracking         (t columns)
      → absolutely closest non-NaN observation
"""

from __future__ import annotations

import math
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import (
    EPS1_BPS,
    EPS2,
    EPS3,
    EPS4,
    T_SCHEDULING_LAG,
    T_SETTLEMENT_LAG,
    DEFAULT_N_SIMS,
    COL_DATED_BRENT,
    COL_WTI_HOUSTON,
    COL_T_SEA_PASSAGE,
    COL_T_ORIGIN_BERTH,
    COL_T_ORIGIN_PORT,
    COL_T_DEST_BERTH,
    COL_WS_QUOTE,
    COL_TD25_FLAT,
    COL_SOFR,
    COL_FX,
)
from .pnl import compute_pnl
from .t1_selector import T1Selector


class BlockOverrunError(Exception):
    """
    Raised when a simulated block extends past the end of the market data
    matrix.  Overruns are expected for blocks starting close to the matrix
    end and are excluded from the P&L distribution by the outer loop, which
    retries until the target simulation count is reached.

    Carries t1_date so the outer loop can record per-date overrun counts for
    diagnostic reporting.
    """

    def __init__(self, message: str, t1_date: pd.Timestamp | None = None) -> None:
        super().__init__(message)
        self.t1_date = t1_date


# ── Triangular sampler ────────────────────────────────────────────────────────

def _tri(rng: np.random.Generator, params: tuple) -> float:
    """Sample from Tri(min, max, mode). params = (min, max, mode)."""
    lo, hi, mode = params
    return float(rng.triangular(lo, mode, hi))


# ── Matrix reading helpers (§3.5.5) ──────────────────────────────────────────

def _read_financial_single(
    series: pd.Series,
    target: pd.Timestamp,
) -> float:
    """
    §3.5.5 — single-input financial value (SOFR, FX, WS Quote, TD25 Flat Rate).
    Returns the nearest previous non-NaN assessment at or before `target`.
    """
    prior = series.loc[:target].dropna()
    if prior.empty:
        raise ValueError(
            f"No non-NaN value for '{series.name}' at or before {target.date()}."
        )
    return float(prior.iloc[-1])


def _read_financial_window(
    series: pd.Series,
    target: pd.Timestamp,
) -> float:
    """
    §3.5.5 — 5-day pricing window (WTI Houston FOB, Dated Brent).

    Anchors at the nearest non-NaN assessment at or before `target`, then takes
    the two nearest trading-day values on each side (positions −2, −1, 0, +1, +2
    in the non-NaN series). Returns the arithmetic mean of the window.

    Boundary behaviour: if fewer than 5 assessments exist around the anchor,
    the available subset is averaged (window is clamped, not padded).
    """
    non_nan = series.dropna()
    if non_nan.empty:
        raise ValueError(f"No non-NaN data for '{series.name}'.")

    # Anchor: last non-NaN at or before target
    prior = non_nan.loc[:target]
    if prior.empty:
        raise ValueError(
            f"No non-NaN value for '{series.name}' at or before {target.date()}."
        )
    anchor_pos = non_nan.index.get_loc(prior.index[-1])

    lo = max(0, anchor_pos - 2)
    hi = min(len(non_nan) - 1, anchor_pos + 2)
    return float(non_nan.iloc[lo : hi + 1].mean())


def _read_vessel_closest(
    series: pd.Series,
    target: pd.Timestamp,
) -> float:
    """
    §3.5.5 — vessel tracking data (t columns).
    Returns the value at the absolutely closest non-NaN observation to `target`.
    """
    non_nan = series.dropna()
    if non_nan.empty:
        raise ValueError(f"No vessel tracking data for '{series.name}'.")
    # pandas 2.x: TimedeltaIndex.abs() removed; use int64 nanoseconds
    abs_ns = np.abs((non_nan.index - target).asi8)
    return float(non_nan.iloc[abs_ns.argmin()])


# ── Parametric draw ───────────────────────────────────────────────────────────

def _draw_parametric(rng: np.random.Generator) -> dict:
    """
    §3.5.2 Step 1 — sample all non-matrix parametric variables.
    """
    return {
        "t_scheduling_lag": _tri(rng, T_SCHEDULING_LAG),
        "t_settlement_lag": _tri(rng, T_SETTLEMENT_LAG),
        "eps1_bps":         _tri(rng, EPS1_BPS),
        "eps2":             _tri(rng, EPS2),
        "eps3":             _tri(rng, EPS3),
        "eps4":             _tri(rng, EPS4),
    }


# ── Date arithmetic ───────────────────────────────────────────────────────────

def _advance(base: pd.Timestamp, days: float) -> pd.Timestamp:
    """Add a (possibly fractional) number of days to a Timestamp."""
    return base + timedelta(days=float(days))


# ── Inner loop ────────────────────────────────────────────────────────────────

def inner_loop(
    matrix: pd.DataFrame,
    wti_level: float,
    spread: float,
    t1_selector: T1Selector,
    rng: np.random.Generator,
) -> dict:
    """
    §3.5.6 — single P&L sample.

    Traces a contiguous block through the market data matrix starting at t1,
    reads the required values at each of the six nodes, and evaluates the P&L
    function. Returns a flat dict of outputs for the outer loop to collect.

    Parameters
    ----------
    matrix      : Market data matrix (DatetimeIndex, ascending).
    wti_level   : Current WTI Houston FOB ($/bbl), live input at t=0.
    spread      : Current Brent–WTI spread ($/bbl), live input at t=0.
    t1_selector : T1Selector instance.
    rng         : NumPy random generator.

    Node timeline (§3.5.3)
    ----------------------
    Node 1 : t1                                         (arbitrage block start)
    Node 2 : Node 1 + (t_sched − 1)                    (vessel at origin port limits)
    Node 3 : Node 2 + t_origin_port_arr→port_dep        (BL date)
    Node 4 : Node 3 + t_sea_passage                     (vessel at dest port limits)
    Node 5 : Node 4 + t_dest_port_arr→berth_dep / 24   (discharge complete)
    Node 6 : Node 5 + t_settlement_lag                  (financing repaid)

    Values read from matrix at t0 = t1 − 1 (§3.4.3, §3.5.2 Table 2, fn. 1)
    -------------------------------------------------------------------------
    WS Quote, TD25 Flat Rate — bootstrapped at the arbitrage decision date
    t0 = t1 − 1, one day before the matrix block starts at t1.
    """

    # ── Step 1: parametric draws ──────────────────────────────────────────────
    p = _draw_parametric(rng)

    # ── Step 2: locate t1 (§3.5.2) ───────────────────────────────────────────
    node1 = t1_selector.select(matrix, wti_level, spread, rng)

    # Convenience: pre-extract non-NaN series for each column
    s = {col: matrix[col] for col in matrix.columns}

    # ── t0: freight fixed at arbitrage decision (t1 − 1) (§3.4.3, Table 2) ──
    # t1 is a next-day forecast starting point; WS Quote and TD25 Flat Rate
    # are read at the arbitrage decision date t0 = t1 − 1, not at t1.
    t0             = _advance(node1, -1)
    ws_quote       = _read_financial_single(s[COL_WS_QUOTE], t0)
    td25_flat_rate = _read_financial_single(s[COL_TD25_FLAT], t0)

    # ── Node 2: vessel arrives at origin port limits ──────────────────────────
    node2 = _advance(node1, p["t_scheduling_lag"] - 1)

    t_origin_port_days = _read_vessel_closest(s[COL_T_ORIGIN_PORT], node2)
    t_origin_berth_hrs = _read_vessel_closest(s[COL_T_ORIGIN_BERTH], node2)

    # ── Node 3: BL date (vessel departs origin port) ──────────────────────────
    node3      = _advance(node2, t_origin_port_days)
    matrix_end = matrix.index[-1]
    if node3 > matrix_end:
        raise BlockOverrunError(
            f"BL date {node3.date()} past matrix end {matrix_end.date()}",
            t1_date=node1,
        )

    t_sea_passage_days = _read_vessel_closest(s[COL_T_SEA_PASSAGE], node3)
    p_wti_5day         = _read_financial_window(s[COL_WTI_HOUSTON], node3)
    fx_bl              = _read_financial_single(s[COL_FX], node3)
    sofr_bl            = _read_financial_single(s[COL_SOFR], node3)

    # ── Node 4: vessel arrives at destination port limits ─────────────────────
    node4 = _advance(node3, t_sea_passage_days)

    t_dest_berth_hrs = _read_vessel_closest(s[COL_T_DEST_BERTH], node4)

    # ── Node 5: discharge complete, sell leg priced ───────────────────────────
    # t_dest_berth_hrs is hours → convert to days for date arithmetic
    node5 = _advance(node4, t_dest_berth_hrs / 24.0)
    if node5 > matrix_end:
        raise BlockOverrunError(
            f"Discharge date {node5.date()} past matrix end {matrix_end.date()}",
            t1_date=node1,
        )

    p_brent_5day  = _read_financial_window(s[COL_DATED_BRENT], node5)
    fx_discharge  = _read_financial_single(s[COL_FX], node5)

    # ── Node 6: settlement (total exposure ends) ──────────────────────────────
    node6 = _advance(node5, p["t_settlement_lag"])

    # ── Derived durations (§3.5.4, eqs. 24-25) ───────────────────────────────
    # §3.3.3: floor() converts continuous accrual to elapsed days (ACT/365)
    financing_exposure_days = math.floor(
        (node6 - node3).total_seconds() / 86_400
    )
    total_exposure_days     = (node6 - node1).total_seconds() / 86_400

    # ── P&L calculation (§3.3) ────────────────────────────────────────────────
    result = compute_pnl(
        p_brent_5day=p_brent_5day,
        p_wti_5day=p_wti_5day,
        eps2=p["eps2"],
        ws_quote=ws_quote,
        td25_flat_rate=td25_flat_rate,
        sofr_bl=sofr_bl,
        eps1_bps=p["eps1_bps"],
        financing_exposure_days=financing_exposure_days,
        t_origin_berth_hrs=t_origin_berth_hrs,
        t_dest_berth_hrs=t_dest_berth_hrs,
        fx_bl=fx_bl,
        fx_discharge=fx_discharge,
        eps3=p["eps3"],
        eps4=p["eps4"],
    )

    result["ws_quote"]                = ws_quote
    result["td25_flat_rate"]          = td25_flat_rate
    result["total_exposure_days"]     = total_exposure_days
    result["financing_exposure_days"] = financing_exposure_days
    result["node1_date"] = node1
    result["node3_date"] = node3   # BL date
    result["node5_date"] = node5   # discharge date
    result["node6_date"] = node6   # settlement date
    result["p_wti_5day"]           = p_wti_5day
    result["eps2"]                 = p["eps2"]
    result["t_sea_passage_days"]   = t_sea_passage_days
    result["t_origin_port_days"]   = t_origin_port_days
    result["t_origin_berth_hrs"]   = t_origin_berth_hrs
    result["t_dest_berth_hrs"]     = t_dest_berth_hrs
    result["sofr_bl"]              = sofr_bl
    result["fx_bl"]                = fx_bl
    result["p_brent_5day"]         = p_brent_5day
    result["t_scheduling_lag"]     = p["t_scheduling_lag"]
    result["t_settlement_lag"]     = p["t_settlement_lag"]
    result["eps1_bps"]             = p["eps1_bps"]

    return result


# ── Outer loop ────────────────────────────────────────────────────────────────

def outer_loop(
    matrix: pd.DataFrame,
    wti_level: float,
    spread: float,
    t1_selector: T1Selector,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Outer simulation loop: iterate inner_loop n_sims times.

    Parameters
    ----------
    matrix      : Market data matrix (DatetimeIndex, ascending).
    wti_level   : Current WTI Houston FOB ($/bbl).
    spread      : Current Brent–WTI spread ($/bbl).
    t1_selector : T1Selector instance (MahalanobisT1Selector).
    n_sims      : Number of Monte-Carlo iterations.
    seed        : Random seed for reproducibility.

    Returns
    -------
    tuple (pd.DataFrame, list[pd.Timestamp]):
        DataFrame with columns:
            pnl, spread, freight, financing, demurrage, insurance, port_fees,
            ws_quote, td25_flat_rate,
            total_exposure_days, financing_exposure_days,
            t_sea_passage_days, t_origin_port_days, t_origin_berth_hrs, t_dest_berth_hrs,
            sofr_bl, fx_bl, p_brent_5day, p_wti_5day,
            node1_date, node3_date, node5_date, node6_date
        overrun_dates: t1 dates of every aborted block (one entry per overrun).
    """

    rng           = np.random.default_rng(seed)
    records: list = []
    overrun_dates: list[pd.Timestamp] = []
    overruns      = 0
    failed        = 0
    attempts      = 0
    max_attempts  = n_sims * 20   # safety cap: tolerate up to 95% overrun rate

    while len(records) < n_sims and attempts < max_attempts:
        attempts += 1
        try:
            row = inner_loop(
                matrix=matrix,
                wti_level=wti_level,
                spread=spread,
                t1_selector=t1_selector,
                rng=rng,
            )
            records.append(row)
        except NotImplementedError:
            raise   # propagate — t1 selector not implemented
        except BlockOverrunError as exc:
            overruns += 1
            if exc.t1_date is not None:
                overrun_dates.append(exc.t1_date)
        except Exception as exc:
            failed += 1
            if failed == 1:
                warnings.warn(
                    f"Inner loop failed on attempt {attempts}: {exc}. "
                    "Subsequent unexpected failures will be silently skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if failed:
        warnings.warn(
            f"{failed:,} unexpected errors excluded from results.",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(records) < n_sims:
        warnings.warn(
            f"Target of {n_sims:,} simulations not reached after {max_attempts:,} "
            f"attempts ({overruns:,} overruns, {failed:,} errors). "
            f"Returning {len(records):,} completed simulations.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not records:
        raise RuntimeError("All simulation iterations failed. Check matrix coverage.")

    return pd.DataFrame(records), overrun_dates
