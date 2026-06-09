"""CRN freight-timing comparison statistics (Common Random Numbers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def run_crn_freight_comparison(sim_df: pd.DataFrame) -> dict:
    """
    Compute CRN freight-timing statistics from outer_loop sim_df.

    All per-path variables other than the freight read date are held constant
    by the inner loop, so delta = pnl_bl - pnl_fixture isolates the freight
    timing effect (BL date vs arbitrage decision date, Node 3).

    Parameters
    ----------
    sim_df : DataFrame returned by outer_loop; must contain 'pnl' and
             'pnl_node3_freight'.

    Returns
    -------
    dict with keys:
        pnl_fixture : np.ndarray  USD — freight priced at arbitrage decision
        pnl_bl      : np.ndarray  USD — freight priced at BL date (Node 3)
        delta       : np.ndarray  USD — pnl_bl minus pnl_fixture
        stats       : dict of summary statistics
    """
    pnl_fixture = sim_df["pnl"].to_numpy(dtype=float)
    pnl_bl      = sim_df["pnl_node3_freight"].to_numpy(dtype=float)
    delta       = pnl_bl - pnl_fixture

    return {
        "pnl_fixture": pnl_fixture,
        "pnl_bl":      pnl_bl,
        "delta":       delta,
        "stats": {
            "mean_delta_usd":   float(delta.mean()),
            "std_delta_usd":    float(delta.std(ddof=1)),
            "skew_delta":       float(stats.skew(delta, bias=False)),
            "p10_delta_usd":    float(np.percentile(delta, 10)),
            "p90_delta_usd":    float(np.percentile(delta, 90)),
            "median_delta_usd": float(np.median(delta)),
            "std_fixture_usd":  float(pnl_fixture.std(ddof=1)),
            "std_bl_usd":       float(pnl_bl.std(ddof=1)),
            "corr_fixture_bl":  float(np.corrcoef(pnl_fixture, pnl_bl)[0, 1]),
            "p_bl_gt_fixture":  float((pnl_bl > pnl_fixture).mean()),
            "n":                int(len(pnl_fixture)),
        },
    }
