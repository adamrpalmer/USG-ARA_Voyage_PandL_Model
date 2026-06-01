"""
Entry point for the FOB USG-NWE Voyage P&L Model.

Usage
-----
python main.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure Unicode output works on Windows terminals (e.g. the π symbol in report.py)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import load_matrix
from src.t1_selector import MahalanobisT1Selector
from src.simulate import outer_loop
from src.report import print_summary, plot_pnl_distribution, plot_audit_diagnostics
from src.config import COL_WTI_HOUSTON, COL_DATED_BRENT

_DATA_PATH = "data/processed/FOB_USG_ARA_Market_Data_Matrix.csv"
_N_SIMS    = 10_000


def _print_admissibility_audit(
    selector: MahalanobisT1Selector,
    wti_level: float,
    spread: float,
    matrix: pd.DataFrame,
    sim_df: pd.DataFrame | None = None,
    overrun_dates: list | None = None,
) -> None:
    """
    Print a two-part diagnostic:
      Part 1 (pre-simulation) — admissible candidate set under the hard
                                 market admissibility filter (θ=5%, eq. 20).
      Part 2 (post-simulation) — distribution of WTI at the actually selected
                                 t1 dates and per-date simulation outcomes.
    """
    diag = selector.diagnostic_admissible_set(wti_level, spread)
    wd   = selector.weight_distribution(wti_level, spread)
    wi        = wd["wi"]
    wti_all   = wd["wti"]
    brent_all = wd["brent"]
    I_market  = wd["I_market"]
    mask_adm  = I_market > 0

    brent_input  = wti_level + spread
    today_doy    = pd.Timestamp.today().timetuple().tm_yday

    W = 72
    print("=" * W)
    print("  t1 Admissibility Diagnostic")
    print("=" * W)

    # ── Bandwidth ────────────────────────────────────────────────────────────
    print(f"  hM bandwidth            : {diag['hm']:>10.4f}  Mahalanobis units")
    print(f"  Hard filter θ           : {selector._THETA:.0%}  relative deviation")
    print()

    # ── Date coverage ────────────────────────────────────────────────────────
    matrix_end = matrix.index[-1]

    print(f"  Matrix end date         : {matrix_end.date()}")
    print(f"  Eligible date range     : {wd['dates'][0].date()}  ->  "
          f"{wd['dates'][-1].date()}  (no tail cutoff)")
    print()

    # ── Top 10 candidates by joint weight wi ─────────────────────────────────
    top10_idx = np.argsort(wi)[::-1][:10]
    have_outcomes = sim_df is not None and overrun_dates is not None

    if have_outcomes:
        from collections import Counter
        retained_ctr = Counter(pd.DatetimeIndex(sim_df["node1_date"]))
        overrun_ctr  = Counter(pd.DatetimeIndex(overrun_dates))

        print("  Top 10 candidates by wi:")
        hdr = (f"    {'':>2}  {'date':<10}  {'WTI':>6}  {'Brent':>6}  "
               f"{'wi':>6}  {'selected_n':>10}  {'aborted_n':>9}  "
               f"{'abort_rate':>10}  {'retained_n':>10}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for rank, idx_t in enumerate(top10_idx, 1):
            d_t        = wd["dates"][idx_t]
            retained_n = retained_ctr.get(d_t, 0)
            aborted_n  = overrun_ctr.get(d_t, 0)
            selected_n = retained_n + aborted_n
            rate       = aborted_n / selected_n * 100 if selected_n else 0.0
            print(f"  {rank:>2}  {str(d_t.date()):<10}  "
                  f"{wd['wti'][idx_t]:>6.2f}  {wd['brent'][idx_t]:>6.2f}  "
                  f"{wi[idx_t]:>6.4f}  {selected_n:>10,}  {aborted_n:>9,}  "
                  f"{rate:>9.1f}%  {retained_n:>10,}")
    else:
        print("  Top 10 candidates by wi:")
        hdr = (f"    {'':>2}  {'date':<10}  {'WTI':>6}  {'Brent':>6}  "
               f"{'d_season':>9}  {'wi':>6}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for rank, idx_t in enumerate(top10_idx, 1):
            d_t     = wd["dates"][idx_t]
            doy_t   = d_t.timetuple().tm_yday
            dseas_t = min(abs(doy_t - today_doy), 365 - abs(doy_t - today_doy))
            print(f"  {rank:>2}  {str(d_t.date()):<10}  "
                  f"{wd['wti'][idx_t]:>6.2f}  {wd['brent'][idx_t]:>6.2f}  "
                  f"{dseas_t:>8d}d  {wi[idx_t]:>6.4f}")
    print()

    # ── Candidate counts ─────────────────────────────────────────────────────
    n_adm    = diag["n_admissible"]
    pct_adm  = n_adm / diag["n_eligible"] * 100 if diag["n_eligible"] else 0.0
    if diag["insufficient"]:
        insuf_detail = "  YES  <-- WARNING  (0 admissible candidates)"
    else:
        insuf_detail = f"  No   ({n_adm:,} admissible candidates > 0)"
    print(f"  Total candidates        : {diag['n_eligible']:>10,}")
    print(f"  Admissible (pi > 0)     : {n_adm:>10,}  ({pct_adm:.1f} %)")
    print(f"  Insufficient candidates :{insuf_detail}")
    print()

    if diag["insufficient"]:
        print("  WARNING: No admissible candidates. Cannot compute distribution stats.")
        print("=" * W)
        print()
        return

    # ── WTI / Brent distribution of the admissible set ───────────────────────
    wti_adm   = wti_all[mask_adm]
    brent_adm = brent_all[mask_adm]

    # pi for admissible set (normalised over all D, subset of non-zero)
    wi_adm    = wi[mask_adm]
    pi_adm    = wi_adm / wi_adm.sum()
    brent_wgt_mean = float(np.dot(pi_adm, brent_adm))
    bias_wgt_wti   = diag["wti_weighted_mean"] - wti_level
    bias_wgt_brent = brent_wgt_mean - brent_input

    print("  WTI / Brent distribution  —  admissible candidates (unweighted percentiles)")
    print(f"    {'':5}  {'WTI':>10}   {'Brent':>10}")
    print(f"    {'Input':5}  ${wti_level:>9.2f}   ${brent_input:>9.2f}")
    print(f"    {'-'*5}  {'-'*10}   {'-'*10}")
    for pct, lbl in [(5, "P5"), (25, "P25"), (50, "P50"), (75, "P75"), (95, "P95")]:
        vw = np.percentile(wti_adm, pct)
        vb = np.percentile(brent_adm, pct)
        print(f"    {lbl:<5}  ${vw:>9.2f}   ${vb:>9.2f}")
    print(f"    {'Mean':<5}  ${wti_adm.mean():>9.2f}   ${brent_adm.mean():>9.2f}")
    print()
    bw_str = f"{'+'if bias_wgt_wti>=0 else ''}{bias_wgt_wti:.2f}"
    bb_str = f"{'+'if bias_wgt_brent>=0 else ''}{bias_wgt_brent:.2f}"
    print(f"  Weighted mean (pi)  WTI  : ${diag['wti_weighted_mean']:>7.2f}/bbl  "
          f"[vs input  ->  {bw_str}/bbl]")
    print(f"  Weighted mean (pi)  Brent: ${brent_wgt_mean:>7.2f}/bbl  "
          f"[vs input  ->  {bb_str}/bbl]")
    print(f"  Max D_M in admissible set: {diag['dm_max_admissible']:>10.4f}")
    print(f"  Date span (admissible)   : {diag['date_min']}  ->  {diag['date_max']}")
    print()

    # ── wi distribution across ALL eligible candidates ───────────────────────
    print("  wi distribution  (all eligible candidates)")
    print(f"    {'Band':<32}  {'n':>6}   {'%':>5}   {'WTI mean':>9}   {'Brent mean':>10}")
    bins = [
        (0.70, 1.01, "> 0.70   (admissible, strong)"),
        (0.30, 0.70, "0.30-0.70  (admissible)      "),
        (0.00, 0.30, "> 0.00-0.30  (admissible, weak)"),
        (-1.0, 0.00, "= 0.00   (excluded by filter)"),
    ]
    for lo, hi, label in bins:
        if lo < 0:
            b_mask = wi == 0.0
        else:
            b_mask = (wi > lo) & (wi < hi)
        n_bin  = int(b_mask.sum())
        pct_b  = n_bin / len(wi) * 100
        if n_bin:
            wmu = wti_all[b_mask].mean()
            bmu = brent_all[b_mask].mean()
            tail = f"  ${wmu:>7.2f}     ${bmu:>7.2f}"
        else:
            tail = ""
        print(f"    {label:<32}  {n_bin:>6,}  ({pct_b:>5.1f}%){tail}")
    print()

    # ── w_market vs w_season decomposition for admissible set ────────────────
    wm_adm = wd["wmarket"][mask_adm]
    ws_adm = wd["wseason"][mask_adm]
    print("  Admissible set  —  weight component means")
    print(f"    w_market (market sim.)  : {wm_adm.mean():.4f}")
    print(f"    w_season (seasonal sim.): {ws_adm.mean():.4f}")
    print(f"    min w_market (admissible): {wm_adm.min():.4f}")
    print()

    # ── Post-simulation: actual t1 WTI / Brent draws ────────────────────────
    if sim_df is not None and "node1_date" in sim_df.columns:
        node1_dates   = pd.DatetimeIndex(sim_df["node1_date"])
        wti_series    = matrix[COL_WTI_HOUSTON]
        brent_series  = matrix[COL_DATED_BRENT]

        wti_at_t1   = wti_series.reindex(node1_dates).values
        brent_at_t1 = brent_series.reindex(node1_dates).values

        if np.isnan(wti_at_t1).any():
            wti_at_t1 = np.array([
                float(wti_series.loc[:d].dropna().iloc[-1])
                for d in node1_dates
            ])
        if np.isnan(brent_at_t1).any():
            brent_at_t1 = np.array([
                float(brent_series.loc[:d].dropna().iloc[-1])
                for d in node1_dates
            ])

        sim_bias_wti   = float(wti_at_t1.mean())  - wti_level
        sim_bias_brent = float(brent_at_t1.mean()) - brent_input

        print(f"  Actual t1 WTI / Brent  (post-simulation,  n = {len(sim_df):,})")
        print(f"    {'':5}  {'WTI':>10}   {'Brent':>10}")
        print(f"    {'Input':5}  ${wti_level:>9.2f}   ${brent_input:>9.2f}")
        print(f"    {'-'*5}  {'-'*10}   {'-'*10}")
        for pct, lbl in [(5, "P5"), (25, "P25"), (50, "P50"), (75, "P75"), (95, "P95")]:
            vw = np.percentile(wti_at_t1, pct)
            vb = np.percentile(brent_at_t1, pct)
            print(f"    {lbl:<5}  ${vw:>9.2f}   ${vb:>9.2f}")
        sbw = f"{'+'if sim_bias_wti>=0 else ''}{sim_bias_wti:.2f}"
        sbb = f"{'+'if sim_bias_brent>=0 else ''}{sim_bias_brent:.2f}"
        print(f"    {'Mean':<5}  ${wti_at_t1.mean():>9.2f}   ${brent_at_t1.mean():>9.2f}"
              f"   [bias: WTI {sbw}  Brent {sbb}]")
        print()

        unique_dates, counts = np.unique(node1_dates, return_counts=True)
        unique_ts = pd.DatetimeIndex(unique_dates)
        top5_idx  = np.argsort(counts)[::-1][:5]
        print(f"  Most-drawn t1 dates  (top 5 of {len(unique_ts):,} unique):")
        for idx in top5_idx:
            ts    = unique_ts[idx]
            cnt   = counts[idx]
            wv    = float(wti_series.loc[ts])   if ts in wti_series.index   else float("nan")
            bv    = float(brent_series.loc[ts]) if ts in brent_series.index else float("nan")
            print(f"    {ts.date()}  WTI ${wv:.2f}  Brent ${bv:.2f}  —  "
                  f"drawn {cnt:,}x  ({cnt/len(sim_df)*100:.1f}%)")

    print("=" * W)
    print()


def main() -> None:
    print("=" * 62)
    print("  FOB USG-NWE Voyage P&L Model")
    print("=" * 62)
    print()
    print("  Simulation Mode")
    print("  [1] Live     — Σ calibrated to recent 3-month volatility")
    print("  [2] Scenario — Σ calibrated to benchmark price levels")
    print()
    while True:
        mode_input = input("  Select mode (1 or 2): ").strip()
        if mode_input in ("1", "2"):
            break
        print("  Please enter 1 or 2.")
    mode = "live" if mode_input == "1" else "scenario"
    print()

    wti    = float(input("  WTI Houston FOB ($/bbl)  : "))
    spread = float(input("  Brent-WTI spread ($/bbl) : "))
    print()

    matrix   = load_matrix(_DATA_PATH)
    selector = MahalanobisT1Selector(matrix, mode=mode)
    cal_info = selector.calibrate(wti, spread)

    print(f"  Σ calibration mode  : {cal_info['mode'].capitalize()}")
    print(f"  Σ calibration obs.  : {cal_info['n_sigma_obs']:,}")
    if "price_range" in cal_info:
        print(f"  Pricing range used  : ±${cal_info['price_range']:.1f}/bbl")
    print()

    ess = selector.compute_ess(wti, spread)

    sim_df, overrun_dates = outer_loop(
        matrix=matrix,
        wti_level=wti,
        spread=spread,
        t1_selector=selector,
        n_sims=_N_SIMS,
    )

    tail_ess = selector.compute_tail_ess(sim_df)

    # Post-simulation: augment audit with actual t1 draws and per-date outcomes
    _print_admissibility_audit(selector, wti, spread, matrix,
                           sim_df=sim_df, overrun_dates=overrun_dates)

    print_summary(sim_df, wti_level=wti, spread=spread, ess=ess, tail_ess=tail_ess,
                  cal_info=cal_info, n_aborted=len(overrun_dates))

    plot_pnl_distribution(sim_df, wti_level=wti, spread=spread, ess=ess,
                          tail_ess=tail_ess, cal_info=cal_info, show=False)

    plot_audit_diagnostics(
        sim_df=sim_df,
        selector=selector,
        matrix=matrix,
        wti_level=wti,
        spread=spread,
        overrun_dates=overrun_dates,
        cal_info=cal_info,
        ess=ess,
        tail_ess=tail_ess,
        n_sims_target=_N_SIMS,
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
