"""
High-resolution 101x13 sweep for the FOB USG-ARA Voyage P&L Model.

Grid: WTI $60.00-$85.00 in $0.25 steps (101 levels)
      spread $3.50-$6.50 in $0.25 steps (13 levels)
      1,313 cells x 10,000 sims = 13,130,000 simulations total.

Crash-safe and idempotent: existing rows are skipped on restart.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_matrix
from src.t1_selector import MahalanobisT1Selector
from src.simulate import outer_loop
from sweeps.crn_stats import run_crn_freight_comparison

_DATA_PATH = "data/processed/FOB_USG_ARA_Market_Data_Matrix.csv"
_OUT_CSV   = Path("sweeps/sweep_hires_raw.csv")

_WTI_LEVELS    = [round(60.0 + 0.25 * i, 2) for i in range(101)]  # 60.00 ... 85.00
_SPREAD_LEVELS = [round(3.50 + 0.25 * i, 2) for i in range(13)]   # 3.50  ... 6.50
_N_SIMS        = 10_000
_TOTAL_CELLS   = len(_WTI_LEVELS) * len(_SPREAD_LEVELS)            # 1313

_COLS = [
    "wti", "spread",
    "n_completed", "n_overruns",
    "ess_d", "tail_ess",
    "n_admissible", "n_eligible",
    "wti_weighted_mean", "brent_weighted_mean",
    "wti_bias_abs", "wti_bias_rel",
    "ev_usd", "cvar_usd", "var_usd", "decision_ratio", "p_loss",
    "mean_delta_usd", "std_delta_usd", "skew_delta",
    "median_delta_usd", "p10_delta_usd", "p90_delta_usd",
    "sigma_calibration_obs", "scenario_price_range",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _seed(wti: float, spread: float) -> int:
    return int(round(wti * 1000)) * 100_000 + int(round(spread * 1000))


def _load_done() -> set[tuple[float, float]]:
    if not _OUT_CSV.exists():
        return set()
    df = pd.read_csv(_OUT_CSV)
    return {(round(float(r.wti), 4), round(float(r.spread), 4)) for r in df.itertuples()}


def _append_row(row: dict) -> None:
    write_header = not _OUT_CSV.exists()
    pd.DataFrame([row], columns=_COLS).to_csv(
        _OUT_CSV, mode="a", header=write_header, index=False,
    )


# ── cell runner ───────────────────────────────────────────────────────────────

def _run_cell(matrix: pd.DataFrame, wti: float, spread: float) -> dict:
    selector = MahalanobisT1Selector(matrix, mode="scenario")
    cal_info = selector.calibrate(wti, spread)

    sim_df, overrun_dates = outer_loop(
        matrix=matrix,
        wti_level=wti,
        spread=spread,
        t1_selector=selector,
        n_sims=_N_SIMS,
        seed=_seed(wti, spread),
    )

    # ── risk metrics ──────────────────────────────────────────────────────────
    pnl      = sim_df["pnl"]
    ev_usd   = float(pnl.mean())
    q05      = float(pnl.quantile(0.05))
    var_usd  = float(-q05)
    tail     = pnl[pnl <= q05]
    cvar_usd = float(-tail.mean()) if not tail.empty else float("nan")
    if math.isnan(cvar_usd) or cvar_usd == 0.0:
        decision_ratio = float("inf")
    else:
        decision_ratio = ev_usd / abs(cvar_usd)
    p_loss = float((pnl < 0).mean())

    # ── selector diagnostics ──────────────────────────────────────────────────
    ess_d    = selector.compute_ess(wti, spread)
    tail_ess = selector.compute_tail_ess(sim_df)

    diag         = selector.diagnostic_admissible_set(wti, spread)
    n_admissible = diag["n_admissible"]
    n_eligible   = diag["n_eligible"]
    wti_wmean    = diag.get("wti_weighted_mean", float("nan"))

    wd       = selector.weight_distribution(wti, spread)
    mask_adm = wd["I_market"] > 0
    wi_adm   = wd["wi"][mask_adm]
    if wi_adm.sum() > 0:
        pi_adm      = wi_adm / wi_adm.sum()
        brent_wmean = float(np.dot(pi_adm, wd["brent"][mask_adm]))
    else:
        brent_wmean = float("nan")

    wti_bias_abs = abs(wti_wmean - wti) if not math.isnan(wti_wmean) else float("nan")
    wti_bias_rel = wti_bias_abs / wti   if not math.isnan(wti_bias_abs) else float("nan")

    # ── CRN freight-timing stats ──────────────────────────────────────────────
    crn = run_crn_freight_comparison(sim_df)
    cs  = crn["stats"]

    return {
        "wti":                   wti,
        "spread":                spread,
        "n_completed":           len(sim_df),
        "n_overruns":            len(overrun_dates),
        "ess_d":                 ess_d,
        "tail_ess":              tail_ess,
        "n_admissible":          n_admissible,
        "n_eligible":            n_eligible,
        "wti_weighted_mean":     wti_wmean,
        "brent_weighted_mean":   brent_wmean,
        "wti_bias_abs":          wti_bias_abs,
        "wti_bias_rel":          wti_bias_rel,
        "ev_usd":                ev_usd,
        "cvar_usd":              cvar_usd,
        "var_usd":               var_usd,
        "decision_ratio":        decision_ratio,
        "p_loss":                p_loss,
        "mean_delta_usd":        cs["mean_delta_usd"],
        "std_delta_usd":         cs["std_delta_usd"],
        "skew_delta":            cs["skew_delta"],
        "median_delta_usd":      cs["median_delta_usd"],
        "p10_delta_usd":         cs["p10_delta_usd"],
        "p90_delta_usd":         cs["p90_delta_usd"],
        "sigma_calibration_obs": cal_info["n_sigma_obs"],
        "scenario_price_range":  cal_info.get("price_range", float("nan")),
    }


# ── entry point ───────────────────────────────────────────────────────────────

def run_sweep() -> pd.DataFrame:
    """Run all pending grid cells and return the completed sweep DataFrame."""
    matrix = load_matrix(_DATA_PATH)
    done   = _load_done()

    n_done    = len(done)
    n_pending = _TOTAL_CELLS - n_done

    print(
        f"FOB USG-ARA Voyage P&L Model -- High-Resolution Sweep\n"
        f"  Grid:   {len(_WTI_LEVELS)} WTI levels  "
        f"(${_WTI_LEVELS[0]:.2f}-${_WTI_LEVELS[-1]:.2f}, $0.25 step)  x  "
        f"{len(_SPREAD_LEVELS)} spread levels  "
        f"(${_SPREAD_LEVELS[0]:.2f}-${_SPREAD_LEVELS[-1]:.2f}, $0.25 step)\n"
        f"  Cells:  {_TOTAL_CELLS} total  |  {n_done} done  |  {n_pending} pending\n"
        f"  Sims:   {_N_SIMS:,} per cell  ({_TOTAL_CELLS * _N_SIMS:,} total)\n"
        f"  Est.:   ~57 h wall time  (scenario-mode t1 selector, 13,130,000 sims)\n",
        flush=True,
    )

    if n_pending == 0:
        print("  All cells complete -- nothing to do.", flush=True)
        return pd.read_csv(_OUT_CSV)

    cell_times: list[float] = []
    cell_idx = 0

    for wti in _WTI_LEVELS:
        for spread in _SPREAD_LEVELS:
            cell_idx += 1
            key = (round(wti, 4), round(spread, 4))
            if key in done:
                continue

            t0  = time.perf_counter()
            row = _run_cell(matrix, wti, spread)
            _append_row(row)
            elapsed = time.perf_counter() - t0
            cell_times.append(elapsed)

            n_run = len(cell_times)
            print(
                f"  [{cell_idx:>4}/{_TOTAL_CELLS}]  "
                f"WTI={wti:.2f}  spread={spread:.2f}  "
                f"n={row['n_completed']:>5}  "
                f"ev={row['ev_usd']/1e6:+.3f}$M  "
                f"p_loss={row['p_loss']*100:.1f}%  "
                f"{elapsed:.1f}s",
                flush=True,
            )

            if n_run % 25 == 0:
                mean_t    = sum(cell_times) / n_run
                n_left    = n_pending - n_run
                eta_h     = mean_t * n_left / 3600
                elapsed_h = sum(cell_times) / 3600
                print(
                    f"  -- checkpoint  {n_done + n_run}/{_TOTAL_CELLS} done  "
                    f"elapsed={elapsed_h:.2f}h  "
                    f"mean/cell={mean_t:.1f}s  "
                    f"ETA~{eta_h:.1f}h --",
                    flush=True,
                )

    return pd.read_csv(_OUT_CSV)


if __name__ == "__main__":
    run_sweep()
