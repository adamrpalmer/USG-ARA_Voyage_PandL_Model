"""
Output generation for the FOB USG-NWE Voyage P&L Model.

Produces:
  - Console summary (EV, CVaR, EV/|CVaR|, component breakdown, trade decision)
  - Voyage P&L report figure: compact summary panel + distribution histogram
  - Decision surface over a (WTI level, spread) grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.signal import find_peaks as _find_peaks
from scipy.ndimage import uniform_filter1d as _smooth

from .config import (
    CVAR_ALPHA, DECISION_THRESHOLD, Q_BL_BBL,
    COL_WTI_HOUSTON, COL_DATED_BRENT,
)


# ── Colour palette ─────────────────────────────────────────────────────────────

_C = dict(
    dark       = "#1a1a2e",
    grey       = "#5a5a6e",
    dimgrey    = "#7a8194",
    ltgrey     = "#b5bcc8",
    bg         = "#f5f7fa",
    panel_l    = "#dce2ed",
    green      = "#1a7a3f",
    red        = "#a81c07",
    amber      = "#c4752a",
    blue       = "#3a5f8a",
    salmon     = "#d45f3c",
    ess_green  = "#2e7a45",   # ESS >= 100 — sufficient support
    ess_amber  = "#a0712a",   # 50 <= ESS < 100 — moderate support
    ess_orange = "#a04020",   # 25 <= ESS < 50  — weak support
    ess_red    = "#8a1c08",   # ESS < 25        — severe concentration
)


def _ess_color(val: float) -> str:
    """Diagnostic colour for ESS / Tail-ESS numeric values."""
    if val >= 100:
        return _C["ess_green"]
    elif val >= 50:
        return _C["ess_amber"]
    elif val >= 25:
        return _C["ess_orange"]
    return _C["ess_red"]

# Density-region colour scheme: 5 muted, institutionally neutral tones (no red/green).
# Red and green carry economic meaning elsewhere; these colours must not imply direction.
_MODE_FILLS = (
    "#b0bdc8",   # M1: blue-grey
    "#d4c9b8",   # M2: warm beige
    "#c4bdd8",   # M3: lavender-grey
    "#ccc4b0",   # M4: warm greige
    "#aab0cc",   # M5: slate-blue
)
_MODE_CENTRES = (
    "#5a7080",   # M1: darker blue-grey  (centre line + label)
    "#8a7060",   # M2: darker warm beige
    "#7060a0",   # M3: darker lavender
    "#887860",   # M4: darker greige
    "#506090",   # M5: darker slate-blue
)
_MODE_BOUND = "#9aa4ae"   # left/right boundary dotted lines (shared, neutral)
_MODE_LINE  = "#2a3040"   # centre dashed line and label — uniform across all modes
_MODE_ALPHA = 0.12        # low opacity — histogram bars remain visually dominant

# Unicode subscript digits for CVaR notation (DejaVu Sans supports these)
_SUB = {0: "₀", 1: "₁", 2: "₂", 3: "₃",
        4: "₄", 5: "₅", 6: "₆", 7: "₇",
        8: "₈", 9: "₉"}
# e.g. "CVaR₀.₀₅" renders as CVaR₀.₀₅
_CVAR_LABEL     = f"CVaR{_SUB[0]}.{_SUB[0]}{_SUB[5]}"    # CVaR₀.₀₅
_CVAR_ABS_LABEL = f"|CVaR{_SUB[0]}.{_SUB[0]}{_SUB[5]}|"  # |CVaR₀.₀₅|


# ── Risk statistics ────────────────────────────────────────────────────────────

def compute_ev(pnl: pd.Series) -> float:
    """Expected profit across all simulations."""
    return float(pnl.mean())


def compute_cvar(pnl: pd.Series, alpha: float = CVAR_ALPHA) -> float:
    """
    CVaR_alpha: mean of the alpha left-tail outcomes.
    Returned as a positive number representing the magnitude of expected loss.
    Returns NaN if the tail is empty.
    """
    threshold = pnl.quantile(alpha)
    tail = pnl[pnl <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def compute_decision_metric(ev: float, cvar: float) -> float:
    """
    EV / |CVaR_alpha| per the decision rule.
    CVaR is a positive loss magnitude; returns inf when CVaR <= 0.
    """
    if np.isnan(cvar) or cvar == 0.0:
        return float("inf")
    return ev / abs(cvar)


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
    ess: float | None = None,
    tail_ess: float | None = None,
    cal_info: dict | None = None,
    n_aborted: int | None = None,
) -> None:
    """Print a structured summary of simulation results to stdout."""
    pnl   = sim_df["pnl"]
    ev    = compute_ev(pnl)
    cvar  = compute_cvar(pnl)
    ratio = compute_decision_metric(ev, cvar)

    w = 64
    print("=" * w)
    print("  FOB USG-NWE Voyage P&L Model — Simulation Summary")
    print("=" * w)
    print(f"  WTI Houston FOB         : ${wti_level:>10.2f} /bbl")
    print(f"  Brent-WTI Spread        : ${spread:>10.2f} /bbl")
    if cal_info is not None:
        print(f"  Sigma calib. mode       : {cal_info['mode'].capitalize():>14}")
        print(f"  Sigma calib. obs. (n)   : {cal_info['n_sigma_obs']:>14,}")
        if "price_range" in cal_info:
            rng_str = f"+-${cal_info['price_range']:.1f}/bbl"
            print(f"  Pricing range used      : {rng_str:>14}")
    print(f"  WS Quote (sim mean)     : {sim_df['ws_quote'].mean():>10.1f} WS")
    print(f"  TD25 Flat Rate (sim)    : ${sim_df['td25_flat_rate'].mean():>10.2f} /mt")
    print(f"  Simulations             : {len(pnl):>10,}")
    if n_aborted is not None:
        total_attempts = len(pnl) + n_aborted
        abort_rate = n_aborted / total_attempts * 100 if total_attempts else 0.0
        print(f"  Aborted paths           : {n_aborted:>10,}  ({abort_rate:.1f}% of attempts)")
    if ess is not None:
        print(f"  Similarity set ESS      : {ess:>10.1f}")
    if tail_ess is not None:
        print(f"  Tail-ESS                : {tail_ess:>10.1f}")
    print("-" * w)
    print(f"  Expected Profit  E[pi]  : ${ev:>12,.0f}")
    print(f"  CVaR 0.05 magnitude     : ${cvar:>12,.0f}  (avg loss, positive)")
    print(f"  EV / |CVaR 0.05|        : {ratio:>12.3f}")
    print("=" * w)
    print()
    print("  P&L Component Breakdown (simulation means):")
    comp_labels = {
        "spread":    "gross cargo margin",
        "freight":   "freight",
        "financing": "financing",
        "demurrage": "demurrage",
        "insurance": "insurance",
        "port_fees": "port fees",
    }
    for col, lbl in comp_labels.items():
        sign = "+" if sim_df[col].mean() >= 0 else ""
        print(f"    {lbl:<20}: {sign}${sim_df[col].mean():>12,.0f}")
    print(f"    {'net P&L':<20}:  ${ev:>12,.0f}")
    print()
    print(f"  Avg total trade duration: {sim_df['total_exposure_days'].mean():.1f} days")
    print(f"  Avg financing window    : {sim_df['financing_exposure_days'].mean():.1f} days")
    print()


# ── Mode detection ─────────────────────────────────────────────────────────────

def _detect_modes(
    pnl_m_values: np.ndarray,
    n_bins: int = 120,
    min_prominence_frac: float = 0.08,
    min_distance_bins: int = 6,
    min_bucket_frac: float = 0.005,
    max_modes: int = 10,
) -> list[dict]:
    """
    Detect material local modes in the P&L distribution.

    Procedure:
      1. Build a 120-bin histogram, lightly smoothed (uniform window=4 bins).
      2. Locate peaks with prominence >= 8% of maximum smoothed density and
         minimum separation of 6 bins.
      3. Assign each peak a region bounded by the valley troughs on either side
         (valley-trough boundaries; regions do not overlap).
      4. Discard regions containing < 0.5% of total simulations.
      5. If more than max_modes remain, keep the largest by mass and re-sort L→R.

    Returns a list of dicts ordered left-to-right by modal centre:
        centre, lo, hi  (float, $M)
        mask            (bool ndarray over input)
        n, frac         (count and share of total simulations)
        label           ("M1", "M2", ...)

    Returns [] when no robust modes are found.
    """
    counts, edges = np.histogram(pnl_m_values, bins=n_bins, density=False)
    centres = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth(counts.astype(float), size=4)

    peaks, _ = _find_peaks(
        smoothed,
        prominence=smoothed.max() * min_prominence_frac,
        distance=min_distance_bins,
    )
    if len(peaks) == 0:
        return []

    # Valley-trough boundaries between adjacent peaks
    boundaries = [float(edges[0])]
    for i in range(len(peaks) - 1):
        seg = smoothed[peaks[i]: peaks[i + 1] + 1]
        trough_idx = peaks[i] + int(np.argmin(seg))
        boundaries.append(float(centres[trough_idx]))
    boundaries.append(float(edges[-1]))

    modes: list[dict] = []
    for i, pk in enumerate(peaks):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (pnl_m_values >= lo) & (pnl_m_values <= hi)
        n_in = int(mask.sum())
        frac = n_in / len(pnl_m_values)
        if frac >= min_bucket_frac:
            modes.append(
                {"centre": float(centres[pk]), "lo": lo, "hi": hi,
                 "mask": mask, "n": n_in, "frac": frac}
            )

    if len(modes) > max_modes:
        modes = sorted(modes, key=lambda m: m["frac"], reverse=True)[:max_modes]
        modes = sorted(modes, key=lambda m: m["centre"])

    for i, m in enumerate(modes):
        m["label"] = f"M{i + 1}"

    return modes


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_pnl(v_m: float) -> str:
    return f"+${v_m:.2f}M" if v_m >= 0 else f"-${abs(v_m):.2f}M"


def _fmt_cost(v_m: float) -> str:
    return f"${abs(v_m):.2f}M"


def _fmt_pnl_num(v_m: float) -> str:
    return f"+{v_m:.2f}" if v_m >= 0 else f"{v_m:.2f}"


def _fmt_cost_num(v_m: float) -> str:
    return f"{abs(v_m):.2f}"


# ── Voyage summary panel ───────────────────────────────────────────────────────

def _draw_voyage_summary(
    ax: plt.Axes,
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
    ess: float | None = None,
    tail_ess: float | None = None,
    cal_info: dict | None = None,
) -> list[dict]:
    """
    Render the voyage summary card on ax.
    Returns detected modal regions so the caller annotates the histogram consistently.
    """
    pnl   = sim_df["pnl"]
    ev    = compute_ev(pnl)
    cvar  = compute_cvar(pnl)
    ratio = compute_decision_metric(ev, cvar)

    pnl_m     = pnl / 1e6
    ev_m      = ev / 1e6
    var_m     = float(pnl_m.quantile(CVAR_ALPHA))
    cvar_m    = cvar / 1e6
    prob_loss = float((pnl < 0).mean()) * 100
    n_sims    = len(pnl)
    ratio_str = f"{ratio:.3f}" if not np.isinf(ratio) else "inf"

    # ── Standard conditional buckets ──────────────────────────────────────────
    q05_raw = float(pnl.quantile(CVAR_ALPHA))
    q95_raw = float(pnl.quantile(1.0 - CVAR_ALPHA))
    mask_worst = (pnl <= q05_raw)
    mask_best  = (pnl >= q95_raw)
    n_worst = int(mask_worst.sum())
    n_best  = int(mask_best.sum())

    _COMPS = ["spread", "freight", "financing", "demurrage", "insurance", "port_fees"]

    def _cond_means(mask_series=None):
        df = sim_df[mask_series] if mask_series is not None else sim_df
        return [float(df[c].mean()) / 1e6 for c in _COMPS]

    means_worst = _cond_means(mask_worst)
    means_avg   = _cond_means()
    means_best  = _cond_means(mask_best)

    # ── Detected modal regions ────────────────────────────────────────────────
    modes      = _detect_modes(pnl_m.values)
    mode_means = [_cond_means(m["mask"]) for m in modes]

    # ── Cargo quantity value impact: −ε₂ × Q_BL × P_Brent  (reporting only) ──
    # Eq. (13): Q_discharge = Q_BL × (1 − ε₂) → quantity delta valued at Brent.
    # This is the ε₂-driven slice already embedded inside the Spread component.
    def _cond_cqv(mask_series=None):
        eps2_s  = sim_df["eps2"]
        brent_s = sim_df["p_brent_5day"]
        if mask_series is not None:
            eps2_s  = eps2_s[mask_series]
            brent_s = brent_s[mask_series]
        return float((-eps2_s * brent_s * Q_BL_BBL).mean()) / 1e6

    cqv_worst     = _cond_cqv(mask_worst)
    cqv_avg       = _cond_cqv()
    cqv_best      = _cond_cqv(mask_best)
    cqv_mode_list = [_cond_cqv(m["mask"]) for m in modes]

    # ── Benchmark spread: (P_Brent − P_WTI) × Q_BL  (pure price leg, no ε₂) ──
    def _cond_bmark(mask_series=None):
        brent_s = sim_df["p_brent_5day"]
        wti_s   = sim_df["p_wti_5day"]
        if mask_series is not None:
            brent_s = brent_s[mask_series]
            wti_s   = wti_s[mask_series]
        return float(((brent_s - wti_s) * Q_BL_BBL).mean()) / 1e6

    bmark_worst     = _cond_bmark(mask_worst)
    bmark_avg       = _cond_bmark()
    bmark_best      = _cond_bmark(mask_best)
    bmark_mode_list = [_cond_bmark(m["mask"]) for m in modes]

    # ── Net P&L conditional means (reporting only) ─────────────────────────────
    def _cond_pnl(mask_series=None):
        df = sim_df[mask_series] if mask_series is not None else sim_df
        return float(df["pnl"].mean()) / 1e6

    pnl_worst     = _cond_pnl(mask_worst)
    pnl_avg       = _cond_pnl()
    pnl_best      = _cond_pnl(mask_best)
    pnl_mode_list = [_cond_pnl(m["mask"]) for m in modes]

    # ── Operational input conditional means (all market matrix variables) ─────
    _OP_COLS = [
        "p_brent_5day", "p_wti_5day",
        "ws_quote", "td25_flat_rate",
        "t_sea_passage_days", "t_origin_port_days",
        "t_origin_berth_hrs", "t_dest_berth_hrs",
        "sofr_bl", "fx_bl",
        "t_settlement_lag", "t_scheduling_lag", "eps1_bps",
        "total_exposure_days", "financing_exposure_days",
    ]

    def _cond_op_means(mask_series=None):
        df = sim_df[mask_series] if mask_series is not None else sim_df
        return [float(df[c].mean()) for c in _OP_COLS]

    op_worst      = _cond_op_means(mask_worst)
    op_avg        = _cond_op_means()
    op_best       = _cond_op_means(mask_best)
    mode_op_means = [_cond_op_means(m["mask"]) for m in modes]

    # ── Column data ───────────────────────────────────────────────────────────
    # Standard columns: worst-5%, all simulations, best-5%. Then modal regions L→R.
    # header   : bold label shown in sub-header row 1
    # qualifier: P&L range or description shown in sub-header row 2
    # meta     : observation count + share, shown in the bucket metadata row
    col_data: list[dict] = [
        {"means": means_worst, "op_means": op_worst,
         "cqv_mean": cqv_worst, "bmark_mean": bmark_worst,
         "pnl_mean": pnl_worst,
         "header": "WORST 5%",
         "qualifier": f"P&L <= {var_m:+.2f}M",
         "meta": f"n={n_worst:,}  ({100 * n_worst / n_sims:.1f}%)"},
        {"means": means_avg, "op_means": op_avg,
         "cqv_mean": cqv_avg, "bmark_mean": bmark_avg,
         "pnl_mean": pnl_avg,
         "header": "All simulations",
         "qualifier": "unconditional mean",
         "meta": f"n={n_sims:,}"},
        {"means": means_best, "op_means": op_best,
         "cqv_mean": cqv_best, "bmark_mean": bmark_best,
         "pnl_mean": pnl_best,
         "header": "BEST 5%",
         "qualifier": f"P&L >= {q95_raw / 1e6:+.2f}M",
         "meta": f"n={n_best:,}  ({100 * n_best / n_sims:.1f}%)"},
    ]
    for m, mm, om, cv, bv, pv in zip(
            modes, mode_means, mode_op_means, cqv_mode_list, bmark_mode_list, pnl_mode_list):
        col_data.append(
            {"means": mm, "op_means": om, "cqv_mean": cv, "bmark_mean": bv,
             "pnl_mean": pv,
             "header": f"{m['label']} @ {m['centre']:+.2f}M",
             "qualifier": f"[{m['lo']:+.2f}, {m['hi']:+.2f}]M",
             "meta": f"n={m['n']:,}  ({100 * m['frac']:.1f}%)",
             "mode_label": m["label"]}
        )

    n_data_cols = len(col_data)

    # ── Layout constants (axes-fraction coordinates) ──────────────────────────
    # Exposure block is right-most; data columns must stay left of EXPO_LBL_X.
    METRIC_LBL_X = 0.01    # RISK METRICS label left-align
    METRIC_VAL_X = 0.13    # RISK METRICS value right-align
    COMP_LBL_X   = 0.15    # component name left-align
    DATA_X_START = 0.305   # first data column right-align edge
    DATA_X_END   = 0.97    # last data column right-align edge

    data_xs = list(np.linspace(DATA_X_START, DATA_X_END, n_data_cols))

    # Shadow module-level _C — neutralise blue-grey tones to match Diagnostics aesthetic
    _C = {**globals()["_C"],
        "bg":        "#FFFFFF",
        "panel_l":   "#F0F0F0",
        "grey":      "#444444",
        "dimgrey":   "#888888",
        "ltgrey":    "#CCCCCC",
        "dark":      "#1A1A1A",
    }

    # ── Panel styling ─────────────────────────────────────────────────────────
    ax.set_facecolor(_C["bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#999999")

    t = ax.transAxes

    # Subtle card behind the five decision metric rows only (not ESS rows below)
    ax.add_patch(mpatches.Rectangle(
        (0.0, 0.648), 0.143, 0.296,
        transform=t, facecolor=_C["panel_l"],
        edgecolor="#999999", linewidth=0.3,
        zorder=0, clip_on=False,
    ))
    # Subtle tint behind column header band
    ax.add_patch(mpatches.Rectangle(
        (COMP_LBL_X, 0.878),
        DATA_X_END + 0.010 - COMP_LBL_X, 0.070,
        transform=t, facecolor=_C["panel_l"],
        edgecolor="none", zorder=0, clip_on=False,
    ))

    ev_color    = _C["green"] if ev_m >= 0 else _C["red"]
    var_color   = _C["red"]   if var_m < 0 else _C["green"]
    ratio_color = _C["green"] if ratio >= DECISION_THRESHOLD else _C["red"]
    ploss_color = (
        _C["red"]   if prob_loss > 50
        else _C["amber"] if prob_loss > 30
        else _C["dark"]
    )

    # ── Header (single row) ───────────────────────────────────────────────────
    # Title and decision ratio share y=0.97; compact stats sit inline between them.
    ax.text(0.01, 0.97,
            "Voyage P&L Summary",
            transform=t, ha="left", va="top",
            fontsize=12, fontweight="bold", color=_C["dark"])
    _spread_sign = "+" if spread >= 0 else "-"
    for _x, _lbl, _val in [
        (0.38, "WTI Houston FOB: ",           f"${wti_level:.2f}/bbl"),
        (0.54, "Starting Brent–WTI spread: ", f"{_spread_sign}${abs(spread):.2f}/bbl"),
        (0.70, "Simulations: ",               f"{n_sims:,}"),
    ]:
        ax.text(_x - 0.002, 0.97, _lbl,
                transform=t, ha="right", va="top",
                fontsize=9.0, color=_C["grey"])
        ax.text(_x + 0.002, 0.97, _val,
                transform=t, ha="left", va="top",
                fontsize=9.0, fontweight="bold", color=_C["dark"])
    for _sx in [0.46, 0.62]:
        ax.text(_sx, 0.97, "|",
                transform=t, ha="center", va="top",
                fontsize=9.0, color=_C["ltgrey"])
    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [0.948, 0.948], transform=t,
            color="#999999", lw=1.0, clip_on=False)

    # ── Section headers ───────────────────────────────────────────────────────
    HDR_Y = 0.934
    ax.text(METRIC_LBL_X, HDR_Y, "RISK METRICS",
            transform=t, ha="left", va="top",
            fontsize=8, fontweight="bold", color=_C["dark"])
    ax.text(COMP_LBL_X, HDR_Y,
            "Means of P&L Drivers by Outcome Bucket",
            transform=t, ha="left", va="top",
            fontsize=8, fontweight="bold", color=_C["dark"])

    # ── Sub-column headers: 2 lines (bold label + P&L qualifier) ──────────────
    # Font sizes scale down when many modal columns are present so headers fit
    # within the available column width without overlapping adjacent columns.
    SH1_Y = 0.915
    SH2_Y = 0.896
    if n_data_cols <= 5:
        FS_SH1, FS_SH2 = 7.5, 6.5
    elif n_data_cols <= 7:
        FS_SH1, FS_SH2 = 6.5, 5.5
    else:
        FS_SH1, FS_SH2 = 6.0, 5.0

    for x_right, cd in zip(data_xs, col_data):
        ax.text(x_right, SH1_Y, cd["header"],
                transform=t, ha="right", va="top",
                fontsize=FS_SH1, fontweight="bold", color=_C["dark"])
        ax.text(x_right, SH2_Y, cd["qualifier"],
                transform=t, ha="right", va="top",
                fontsize=FS_SH2, color=_C["grey"])

    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [0.878, 0.878],
            transform=t, color="#999999", lw=0.8, clip_on=False)

    # ── Component metric rows (balance-sheet layout) ──────────────────────────
    # 9 rows: revenue lines → Gross cargo margin subtotal → costs → Net P&L.
    # Thin separator lines are drawn above the two margin rows.
    ROWS_Y = [0.858, 0.818, 0.774, 0.734, 0.694, 0.654, 0.614, 0.574, 0.526]
    FS_LBL = 8.0
    FS_COMP = 8.0
    FS_EXPO = 8.0

    # DECISION METRICS block
    cvar_color = _C["red"] if cvar_m > 0 else _C["green"]
    metric_rows = [
        ("Expected P&L",           _fmt_pnl(ev_m),      ev_color),
        ("VaR 5%",                 _fmt_pnl(var_m),      var_color),
        (_CVAR_LABEL,              _fmt_pnl(-cvar_m),    cvar_color),
        ("P(Loss)",                f"{prob_loss:.1f}%",  ploss_color),
        (f"EV / {_CVAR_ABS_LABEL}", ratio_str,            ratio_color),
    ]
    for y, (lbl, val, col) in zip(ROWS_Y, metric_rows):
        ax.text(METRIC_LBL_X, y, lbl,
                transform=t, ha="left", va="top",
                fontsize=FS_LBL, color=_C["dark"])
        ax.text(METRIC_VAL_X, y, val,
                transform=t, ha="right", va="top",
                fontsize=FS_LBL, fontweight="bold", color=col)

    # Separator lines above the two margin rows
    SEP1_Y = (ROWS_Y[1] + ROWS_Y[2]) / 2   # above Gross cargo margin
    SEP2_Y = (ROWS_Y[7] + ROWS_Y[8]) / 2   # above Net P&L
    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [SEP1_Y, SEP1_Y],
            transform=t, color="#999999", lw=1.0, clip_on=False)
    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [SEP2_Y, SEP2_Y],
            transform=t, color="#555555", lw=1.5, clip_on=False)

    # Component attribution block (balance-sheet order).
    # Each entry: (label, get_value_fn, is_signed, is_margin)
    comp_spec = [
        ("Spread value @ BL qty, $M",      lambda cd: cd["bmark_mean"], True,  False),
        ("Qty adj. @ discharge price, $M", lambda cd: cd["cqv_mean"],   True,  False),
        ("Gross cargo margin, $M",         lambda cd: cd["means"][0],   True,  True),
        ("Freight, $M",                    lambda cd: cd["means"][1],   False, False),
        ("Financing, $M",                  lambda cd: cd["means"][2],   False, False),
        ("Demurrage, $M",                  lambda cd: cd["means"][3],   False, False),
        ("Insurance, $M",                  lambda cd: cd["means"][4],   False, False),
        ("Port Fees, $M",                  lambda cd: cd["means"][5],   False, False),
        ("Net P&L, $M",                    lambda cd: cd["pnl_mean"],   True,  True),
    ]
    for y, (lbl, get_val, signed, is_margin) in zip(ROWS_Y, comp_spec):
        lbl_fw = "bold" if is_margin else "normal"
        ax.text(COMP_LBL_X, y, lbl,
                transform=t, ha="left", va="top",
                fontsize=FS_COMP, fontweight=lbl_fw, color=_C["dark"])
        for x_right, cd in zip(data_xs, col_data):
            v   = get_val(cd)
            txt = _fmt_pnl_num(v) if signed else _fmt_cost_num(v)
            col = (_C["green"] if v >= 0 else _C["red"]) if signed else _C["dark"]
            ax.text(x_right, y, txt,
                    transform=t, ha="right", va="top",
                    fontsize=FS_COMP, fontweight="bold", color=col)


    # ── Operational input rows (WS Quote, TD25 Flat Rate) ────────────────────
    # Thin sub-separator + "MARKET INPUTS" label divides P&L components above
    # from market-rate conditional means below.
    OP_SEP_Y = 0.498
    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [OP_SEP_Y, OP_SEP_Y],
            transform=t, color="#555555", lw=0.8, clip_on=False)
    ax.text(COMP_LBL_X, OP_SEP_Y - 0.005, "P&L INPUTS",
            transform=t, ha="left", va="top",
            fontsize=7, fontweight="bold", color=_C["dark"])

    # 15 rows at 0.030 step.
    OP_ROWS_Y = [0.476, 0.448, 0.420, 0.392, 0.364, 0.336, 0.308, 0.280, 0.252, 0.224, 0.196, 0.168, 0.140, 0.112, 0.084]
    op_row_defs = [
        ("Brent 5D avg, $/bbl",   lambda v: f"{v:.2f}"),
        ("WTI 5D avg, $/bbl",     lambda v: f"{v:.2f}"),
        ("WS quote, WS",          lambda v: f"{v:.1f}"),
        ("TD25 flat, $/mt",       lambda v: f"{v:.3f}"),
        ("Sea passage, days",     lambda v: f"{v:.1f}"),
        ("Orig. port, days",      lambda v: f"{v:.1f}"),
        ("Orig. berth, hrs",      lambda v: f"{v:.1f}"),
        ("Dest. berth, hrs",      lambda v: f"{v:.1f}"),
        ("SOFR, %",               lambda v: f"{v * 100:.3f}"),
        ("FX USD/EUR",            lambda v: f"{v:.4f}"),
        ("Settlement lag, days",  lambda v: f"{v:.1f}"),
        ("Scheduling lag, days",  lambda v: f"{v:.1f}"),
        ("Credit spread, bps",    lambda v: f"{v:.0f}"),
        ("Trade duration, days",  lambda v: f"{v:.1f}"),
        ("Financing window, days", lambda v: f"{v:.1f}"),
    ]
    FS_OP = 8.0
    for j, (op_y, (op_lbl, op_fmt)) in enumerate(zip(OP_ROWS_Y, op_row_defs)):
        ax.text(COMP_LBL_X, op_y, op_lbl,
                transform=t, ha="left", va="top",
                fontsize=FS_OP, color=_C["dark"])
        for x_right, cd in zip(data_xs, col_data):
            v = cd["op_means"][j]
            ax.text(x_right, op_y, op_fmt(v),
                    transform=t, ha="right", va="top",
                    fontsize=FS_OP, fontweight="bold", color=_C["dark"])

    # Closing rule below the last P&L Inputs row
    ax.plot([COMP_LBL_X, data_xs[-1] + 0.01], [0.056, 0.056],
            transform=t, color="#999999", lw=0.6, clip_on=False)

    return modes


# ── P&L distribution plot ──────────────────────────────────────────────────────

def _stagger_label_heights(
    centres: list[float],
    pnl_range: float,
    base_y: float = 1.015,
    step: float   = 0.050,
) -> list[float]:
    """
    Assign staggered y-positions (above the plot area) for mode labels.
    Adjacent labels whose modal centres are within 12% of the P&L range are
    placed at alternating heights to prevent text collision.
    """
    thresh = max(0.6, pnl_range * 0.12)
    heights = []
    for i, c in enumerate(centres):
        if i == 0:
            heights.append(base_y)
        else:
            if abs(c - centres[i - 1]) < thresh:
                heights.append(
                    base_y + step if heights[-1] == base_y else base_y
                )
            else:
                heights.append(base_y)
    return heights


def _build_histogram_figure(
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
    modes: list[dict],
) -> plt.Figure:
    """
    Page 2: standalone P&L distribution histogram.
    Modes are passed in from _draw_voyage_summary so annotations are consistent
    with the table on page 1.
    """
    pnl_m  = sim_df["pnl"] / 1e6
    ev_m   = float(pnl_m.mean())
    var_m  = float(pnl_m.quantile(CVAR_ALPHA))
    tail   = pnl_m[pnl_m <= var_m]
    cvar_m = float(-tail.mean()) if not tail.empty else float("nan")

    fig, ax_hist = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.22)

    ax_hist.set_facecolor(_C["bg"])

    ax_hist.hist(
        pnl_m, bins=100,
        color=_C["blue"], alpha=0.76, edgecolor="none",
        density=True, zorder=3,
    )

    # CVaR left-tail shading
    x_lo = float(pnl_m.min()) - 0.05
    cvar_label = (
        f"CVaR 5% tail  (−${cvar_m:.2f}M avg. loss)"
        if not np.isnan(cvar_m) else "CVaR 5% tail"
    )
    ax_hist.axvspan(x_lo, var_m, alpha=0.20, color="#bf5070",
                    zorder=1, label=cvar_label)

    # Modal region visualisation
    blended = ax_hist.get_xaxis_transform()

    if modes:
        pnl_range = float(pnl_m.max() - pnl_m.min())
        label_ys  = _stagger_label_heights([m["centre"] for m in modes], pnl_range)

        for i, (m, lbl_y) in enumerate(zip(modes, label_ys)):
            fill_color = _MODE_FILLS[i % len(_MODE_FILLS)]
            mode_tag   = f"{m['label']} @ {m['centre']:+.2f}M"

            ax_hist.axvspan(
                m["lo"], m["hi"],
                alpha=_MODE_ALPHA, color=fill_color,
                zorder=2, linewidth=0,
            )
            ax_hist.axvline(m["lo"],     color=_MODE_BOUND, lw=0.5, ls=":", zorder=4)
            ax_hist.axvline(m["hi"],     color=_MODE_BOUND, lw=0.5, ls=":", zorder=4)
            ax_hist.axvline(m["centre"], color=_MODE_LINE,  lw=1.0, ls="--", zorder=4)
            ax_hist.text(
                m["centre"], lbl_y, mode_tag,
                transform=blended, ha="center", va="bottom",
                fontsize=7.0, color=_C["dark"], fontweight="normal",
                clip_on=False,
            )

        if len(modes) > 1:
            mode_legend_label = (
                f"Density regions around detected modes "
                f"({modes[0]['label']}–{modes[-1]['label']})"
            )
        else:
            mode_legend_label = (
                f"Density region around detected mode ({modes[0]['label']})"
            )

        mode_patch = mpatches.Patch(
            facecolor=_MODE_FILLS[0], alpha=0.45,
            edgecolor=_MODE_CENTRES[0], linewidth=0.9,
            label=mode_legend_label,
        )

    # Key reference lines
    ax_hist.axvline(ev_m,  color=_C["green"], lw=1.8,  ls="--", zorder=5,
                    label=f"Expected P&L  {ev_m:+.2f}M")
    ax_hist.axvline(var_m, color=_C["amber"], lw=1.4,  ls="--", zorder=5,
                    label=f"VaR 5%  {var_m:+.2f}M")

    # Legend — anchored to figure bottom so it clears the decile axis
    handles, labels = ax_hist.get_legend_handles_labels()
    if modes:
        handles = [mode_patch] + handles
        labels  = [mode_legend_label] + labels

    legend = ax_hist.legend(
        handles=handles, labels=labels,
        frameon=True, framealpha=0.92, fontsize=8,
        edgecolor=_C["ltgrey"], facecolor=_C["bg"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        ncol=3,
        borderpad=0.6,
        columnspacing=1.5,
    )
    legend.get_frame().set_linewidth(0.5)

    # Axes formatting
    ax_hist.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax_hist.set_xlabel("Voyage P&L (USD millions)", fontsize=10,
                       color=_C["grey"], labelpad=8)
    ax_hist.set_ylabel("Probability Density", fontsize=10,
                       color=_C["grey"], labelpad=8)
    # Title placed in figure-fraction coordinates (above the axes top) so it
    # cannot overlap with mode labels, which use the blended axes transform.
    fig.text(0.07, 0.975, "P&L Distribution",
             fontsize=12, fontweight="bold", color=_C["dark"],
             va="top", ha="left")

    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_color(_C["ltgrey"])
    ax_hist.spines["bottom"].set_color(_C["ltgrey"])
    ax_hist.tick_params(axis="both", colors=_C["grey"], labelsize=9)
    ax_hist.yaxis.grid(True, color="#d8dde4", lw=0.4, zorder=0)
    ax_hist.set_axisbelow(True)

    # P&L decile secondary axis (D1 = P10, D5 = median, D9 = P90)
    decile_probs = np.arange(0.1, 1.0, 0.1)
    decile_vals  = [float(pnl_m.quantile(q)) for q in decile_probs]
    ax_dec = ax_hist.twiny()
    ax_dec.set_xlim(ax_hist.get_xlim())
    ax_dec.set_xticks(decile_vals)
    ax_dec.set_xticklabels(
        [f"D{int(round(q * 10))}" for q in decile_probs],
        fontsize=7.5, color=_C["dimgrey"],
    )
    ax_dec.set_xlabel(
        "P&L deciles  (D1 = P10, D5 = median, D9 = P90)",
        fontsize=7.5, color=_C["dimgrey"], labelpad=4,
    )
    ax_dec.tick_params(
        axis="x", direction="out", length=4,
        colors=_C["dimgrey"], top=False, bottom=True,
        labelbottom=True, labeltop=False,
    )
    ax_dec.xaxis.set_ticks_position("bottom")
    ax_dec.xaxis.set_label_position("bottom")
    # Increased outward offset so decile labels sit below main tick labels without
    # crowding the main xlabel (matplotlib places the main xlabel below the full
    # axes bounding box, which expands to include this outward axis automatically).
    ax_dec.spines["bottom"].set_position(("outward", 50))
    ax_dec.spines["bottom"].set_color(_C["dimgrey"])
    ax_dec.spines["bottom"].set_linewidth(0.6)
    for _sp in ["top", "left", "right"]:
        ax_dec.spines[_sp].set_visible(False)

    return fig


def plot_pnl_distribution(
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
    ess: float | None = None,
    tail_ess: float | None = None,
    cal_info: dict | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Two-page voyage P&L report.

    Page 1 — voyage summary card: decision metrics, conditional component means
              table, exposure block.
    Page 2 — P&L distribution histogram with modal region annotations.

    Modal regions are detected once and applied consistently to both pages so
    that M-labels in the table match those on the histogram.

    If save_path is provided the output is written as a two-page PDF.
    Both figure objects are returned as a tuple (fig_summary, fig_histogram).
    """
    # Page 1: summary card — taller to accommodate 3-row cargo margin decomposition
    fig1 = plt.figure(figsize=(14, 11))
    fig1.patch.set_facecolor("white")
    ax_sum = fig1.add_axes([0.01, 0.02, 0.98, 0.97])
    modes = _draw_voyage_summary(ax_sum, sim_df, wti_level, spread,
                                 ess=ess, tail_ess=tail_ess, cal_info=cal_info)

    # Page 2: histogram (receives modes so labels are consistent)
    fig2 = _build_histogram_figure(sim_df, wti_level, spread, modes)

    if save_path:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(save_path) as pdf:
            pdf.savefig(fig1, bbox_inches="tight")
            pdf.savefig(fig2, bbox_inches="tight")

    if show:
        plt.show()

    return fig1, fig2


# ── Bootstrap stability helper ─────────────────────────────────────────────────

def _bootstrap_stability(
    sim_df: pd.DataFrame,
    selector,
    alpha: float = 0.05,
    n_boot: int  = 200,
    seed: int    = 42,
) -> dict:
    """
    Resample sim_df n_boot times and compute P&L risk metrics + Tail-ESS
    for each resample. Returns dict of numpy arrays (one entry per metric).
    ESS_D is a pre-simulation property; only Tail-ESS is resampled.
    """
    rng = np.random.default_rng(seed)
    n   = len(sim_df)
    out = {k: np.empty(n_boot) for k in
           ("ev", "var", "cvar", "ratio", "ploss", "tail_ess", "unique_t1")}
    for b in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        boot   = sim_df.iloc[idx]
        pnl_b  = boot["pnl"]
        ev_b   = float(pnl_b.mean())
        var_b  = float(pnl_b.quantile(alpha))
        tail_b = pnl_b[pnl_b <= var_b]
        cvar_b = float(-tail_b.mean()) if not tail_b.empty else float("nan")
        if np.isnan(cvar_b) or cvar_b == 0.0:
            ratio_b = float("nan")
        else:
            ratio_b = ev_b / abs(cvar_b)
        ploss_b = float((pnl_b < 0).mean()) * 100
        tess_b  = (selector.compute_tail_ess(boot)
                   if hasattr(selector, "compute_tail_ess") else float("nan"))
        uniq_b  = int(pd.DatetimeIndex(boot["node1_date"]).nunique())
        out["ev"][b]        = ev_b
        out["var"][b]       = var_b
        out["cvar"][b]      = cvar_b
        out["ratio"][b]     = ratio_b
        out["ploss"][b]     = ploss_b
        out["tail_ess"][b]  = tess_b
        out["unique_t1"][b] = uniq_b
    return out


# ── Audit Diagnostics figure ───────────────────────────────────────────────────

def _build_audit_figure(
    sim_df: pd.DataFrame,
    selector,
    matrix: pd.DataFrame,
    wti_level: float,
    spread: float,
    overrun_dates: list,
    cal_info: dict,
    ess: float | None,
    tail_ess: float | None,
    n_sims_target: int,
) -> plt.Figure:
    """Construct and return the Audit Diagnostics figure."""
    from collections import Counter

    brent_level = wti_level + spread
    pnl         = sim_df["pnl"]
    n_retained  = len(sim_df)
    n_aborted   = len(overrun_dates)
    n_attempted = n_retained + n_aborted
    abort_rate  = n_aborted / n_attempted * 100 if n_attempted else 0.0

    # ── Weight distribution ───────────────────────────────────────────────────
    wd        = selector.weight_distribution(wti_level, spread)
    diag      = selector.diagnostic_admissible_set(wti_level, spread)
    wi        = wd["wi"]
    wti_all   = wd["wti"]
    brent_all = wd["brent"]
    I_market  = wd["I_market"]
    mask_adm  = I_market > 0
    dm_all    = wd["dm"]

    n_eligible   = len(wi)
    n_admissible = int(mask_adm.sum())
    pct_adm      = n_admissible / n_eligible * 100 if n_eligible else 0.0
    insufficient = diag["insufficient"]

    wi_adm    = wi[mask_adm]
    wti_adm   = wti_all[mask_adm]
    brent_adm = brent_all[mask_adm]
    wm_adm    = wd["wmarket"][mask_adm]
    ws_adm    = wd["wseason"][mask_adm]

    pi_adm         = wi_adm / wi_adm.sum() if wi_adm.sum() > 0 else np.zeros_like(wi_adm)
    brent_wgt_mean = (float(np.dot(pi_adm, brent_adm))
                      if wi_adm.sum() > 0 else float("nan"))
    wti_wgt_mean   = float(diag.get("wti_weighted_mean", float("nan")))
    bias_wti       = (wti_wgt_mean   - wti_level   if not np.isnan(wti_wgt_mean)   else float("nan"))
    bias_brent     = (brent_wgt_mean - brent_level if not np.isnan(brent_wgt_mean) else float("nan"))

    # ── Post-simulation t1 lookups ────────────────────────────────────────────
    node1_dates  = pd.DatetimeIndex(sim_df["node1_date"])
    wti_series   = matrix[COL_WTI_HOUSTON]
    brent_series = matrix[COL_DATED_BRENT]

    def _lookup(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
        vals = series.reindex(dates).values.astype(float)
        nan_m = np.isnan(vals)
        if nan_m.any():
            for k, d in enumerate(dates):
                if nan_m[k]:
                    prior = series.loc[:d].dropna()
                    vals[k] = float(prior.iloc[-1]) if not prior.empty else float("nan")
        return vals

    wti_at_t1   = _lookup(wti_series,   node1_dates)
    brent_at_t1 = _lookup(brent_series, node1_dates)

    # Unique t1 concentration stats
    unique_dates, counts = np.unique(node1_dates, return_counts=True)
    n_unique_t1  = len(unique_dates)
    draw_shares  = counts / n_retained
    hhi          = float(np.sum(draw_shares ** 2))
    eff_n        = 1.0 / hhi if hhi > 0 else float("nan")
    sort_idx     = np.argsort(counts)[::-1]
    top5_share   = float(counts[sort_idx[:min(5,  n_unique_t1)]].sum() / n_retained * 100)
    top10_share  = float(counts[sort_idx[:min(10, n_unique_t1)]].sum() / n_retained * 100)

    # ── Aborted path WTI / Brent ──────────────────────────────────────────────
    if overrun_dates:
        ovr_ts    = pd.DatetimeIndex(overrun_dates)
        wti_ovr   = _lookup(wti_series,   ovr_ts)
        brent_ovr = _lookup(brent_series, ovr_ts)
        wti_ovr   = wti_ovr[~np.isnan(wti_ovr)]
        brent_ovr = brent_ovr[~np.isnan(brent_ovr)]
    else:
        wti_ovr   = np.array([])
        brent_ovr = np.array([])

    # ── Top 10 candidates by wi ───────────────────────────────────────────────
    top10_wi_idx = np.argsort(wi)[::-1][:10]
    retained_ctr = Counter(pd.DatetimeIndex(sim_df["node1_date"]))
    overrun_ctr  = Counter(pd.DatetimeIndex(overrun_dates)) if overrun_dates else Counter()

    # ── Headline P&L ──────────────────────────────────────────────────────────
    ev        = compute_ev(pnl)
    cvar      = compute_cvar(pnl)
    ratio     = compute_decision_metric(ev, cvar)
    var_05    = float(pnl.quantile(CVAR_ALPHA))
    prob_loss = float((pnl < 0).mean()) * 100

    # ── Bootstrap stability (200 resamples) ───────────────────────────────────
    boot = _bootstrap_stability(sim_df, selector, n_boot=200)

    # ── Mode Provenance diagnostics ───────────────────────────────────────────
    _mp_modes = _detect_modes((sim_df["pnl"] / 1e6).values)
    _mp_t1    = pd.DatetimeIndex(sim_df["node1_date"])

    def _fmt_p(s, e):
        if s == e:
            return s.strftime("%b %Y")
        if s.year == e.year:
            return f"{s.strftime('%b')}–{e.strftime('%b %Y')}"
        return f"{s.strftime('%b %Y')}–{e.strftime('%b %Y')}"

    def _dominant_periods(mask):
        dates = _mp_t1[mask]
        n_in  = int(mask.sum())
        if n_in == 0:
            return "—"
        ym  = dates.to_period("M")
        cts = pd.Series(ym).value_counts()
        n_p = len(cts)
        thr = max(0.10, 2.0 / n_p)
        flagged = cts[cts / n_in > thr].sort_values(ascending=False)
        if flagged.empty:
            return "Broadly distributed"
        periods = sorted(flagged.index.tolist())
        ranges, s, e = [], periods[0], periods[0]
        for p in periods[1:]:
            if p == e + 1:
                e = p
            else:
                ranges.append(_fmt_p(s, e))
                s = e = p
        ranges.append(_fmt_p(s, e))
        out = ";  ".join(ranges[:3])
        return out

    _mp_rows = []
    for _m in _mp_modes:
        _msk = _m["mask"]
        _mp_rows.append({
            "label":   _m["label"],
            "share":   f"{100 * _m['frac']:.1f}%",
            "uniq_t1": int(_mp_t1[_msk].nunique()),
            "periods": _dominant_periods(_msk),
        })

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15), dpi=100)
    fig.patch.set_facecolor("#F2F2F2")
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # ── Body GridSpec: left [1]–[4]  ·  right [5]–[9] ────────────────────────
    # Title sits as fig.text above the grid — no dedicated header row, so the
    # right column reclaims that vertical space for content.
    fig.text(
        0.025, 0.972, "Audit Diagnostics",
        fontsize=12, fontweight="bold", color="#1a1a2e",
        ha="left", va="top",
    )
    gs_body = GridSpec(
        nrows=1, ncols=2,
        figure=fig,
        wspace=0.028,
        left=0.025, right=0.975,
        top=0.948, bottom=0.030,
    )

    # ── Left column — sections [1]–[4] ────────────────────────────────────────
    # height_ratios reflect approx content line counts per section:
    #   [1] Run Config (12)  [2] Admissibility (10)
    #   [3] Market Check (13)  [4] Weight Diagnostics (10)
    gs_left = GridSpecFromSubplotSpec(
        nrows=4, ncols=1,
        subplot_spec=gs_body[0, 0],
        hspace=0.028,
        height_ratios=[12, 10, 13, 10],
    )
    # Section [1] splits horizontally: Run Configuration | Mode Provenance
    _gs_s1 = GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs_left[0], wspace=0.028,
    )
    ax_s1L = fig.add_subplot(_gs_s1[0, 0])  # [1L] Run Configuration
    ax_s1R = fig.add_subplot(_gs_s1[0, 1])  # [1R] Mode Provenance
    ax_s2 = fig.add_subplot(gs_left[1])     # [2] Candidate Admissibility
    ax_s3 = fig.add_subplot(gs_left[2])   # [3] Input-vs-Admissible Market Check
    ax_s4 = fig.add_subplot(gs_left[3])   # [4] Similarity Weight Diagnostics

    # ── Right column — sections [5]–[7], [9] ─────────────────────────────────
    # [8] Headline P&L removed — already present on the P&L Summary page.
    # Freed space redistributed to §6 (densest) and §9.
    gs_right = GridSpecFromSubplotSpec(
        nrows=4, ncols=1,
        subplot_spec=gs_body[0, 1],
        hspace=0.018,
        height_ratios=[12, 23, 15, 16],
    )
    ax_s5 = fig.add_subplot(gs_right[0])  # [5] Top Weighted Candidates
    ax_s6 = fig.add_subplot(gs_right[1])  # [6] Realised t1 Sampling
    ax_s7 = fig.add_subplot(gs_right[2])  # [7] Path Completion / Abort
    ax_s9 = fig.add_subplot(gs_right[3])  # [9] Statistical Reliability

    # ── Audit colour palette ──────────────────────────────────────────────────
    _CARD_BG     = "#FFFFFF"
    _CARD_BORDER = "#999999"   # card outline, 0.8 pt
    _TBL_HDR    = "#F0F0F0"   # table header row fill
    _RULE        = "#CCCCCC"   # table divider rules, 0.5 pt
    # Shadow global _C inside this function — removes all accent colours;
    # the only permitted accent (#1B4F72) would apply to P&L values if present.
    _C = {**globals()["_C"],
        "grey":      "#444444",
        "dimgrey":   "#888888",
        "ltgrey":    "#CCCCCC",
        "green":     "#2C2C2C",
        "red":       "#2C2C2C",
        "amber":     "#2C2C2C",
        "ess_green": "#444444",
        "ess_amber": "#888888",
        "ess_orange": "#888888",
        "ess_red":   "#888888",
    }
    for _cax in (ax_s1L, ax_s1R, ax_s2, ax_s3, ax_s4,
                 ax_s5, ax_s6, ax_s7, ax_s9):
        _cax.set_facecolor(_CARD_BG)
        _cax.set_xticks([])
        _cax.set_yticks([])
        for _sp in _cax.spines.values():
            _sp.set_linewidth(0.8)
            _sp.set_color(_CARD_BORDER)


    # ── Typography constants ──────────────────────────────────────────────────
    _FS = {
        "title":    12.0,   # report title         — matches P&L summary title (12 pt bold)
        "subtitle":  9.0,   # subtitle stats line  — matches P&L summary header stats (9 pt)
        "sec_hdr":   8.0,   # [N] section headings — matches P&L summary "RISK METRICS" (8 pt bold)
        "tbl_hdr":   7.5,   # table column headers — matches P&L summary column headers (7.5 pt bold)
        "tbl_body":  8.0,   # kv labels + row vals — matches P&L summary body rows (8 pt)
        "tbl_sub":   7.5,   # table subtitles      — matches P&L summary secondary rows (7.5 pt)
        "footnote":  7.0,   # footer note          — matches P&L summary footnote (7 pt)
    }

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _hdr(x, y, text, w):
        band_h = 1.8 * STEP
        ax.text(x + 0.004, y - band_h / 2, text, transform=t, ha="left", va="center",
                fontsize=_FS["sec_hdr"], fontweight="bold", color=_C["dark"])
        ax.plot([x - 0.006, x + w + 0.004], [y - band_h, y - band_h],
                transform=t, color=_RULE, lw=0.5, clip_on=False)

    def _kv(xl, xv, y, label, value, vc=None):
        ax.text(xl, y, label, transform=t, ha="left", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(xv, y, str(value), transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], fontweight="bold", color=vc or _C["dark"])

    def _thdr(xs, labels, y, x0, first_left=True):
        ax.add_patch(mpatches.Rectangle(
            (x0, y - STEP), xs[-1] - x0, STEP,
            transform=t, facecolor=_TBL_HDR, edgecolor="none",
            clip_on=False, zorder=0,
        ))
        for i, (x, lbl) in enumerate(zip(xs, labels)):
            if i == 0 and first_left:
                ax.text(x0 + 0.008, y, lbl, transform=t, ha="left", va="top",
                        fontsize=_FS["tbl_hdr"], fontweight="bold", color=_C["dark"])
            else:
                ax.text(x, y, lbl, transform=t, ha="right", va="top",
                        fontsize=_FS["tbl_hdr"], fontweight="bold", color=_C["dark"])
        ax.plot([x0, xs[-1]], [y - STEP, y - STEP],
                transform=t, color=_RULE, lw=0.5, clip_on=False)

    def _trow(xs, vals, y, colors=None, bold=False):
        for i, (x, v) in enumerate(zip(xs, vals)):
            vc = (colors[i] if colors and i < len(colors) else None) or _C["dark"]
            if i == 0:
                ax.text(SLX + 0.008, y, str(v), transform=t, ha="left", va="top",
                        fontsize=_FS["tbl_body"], fontweight="bold" if bold else "normal", color=vc)
            else:
                ax.text(x, y, str(v), transform=t, ha="right", va="top",
                        fontsize=_FS["tbl_body"], fontweight="bold" if bold else "normal", color=vc)

    def _subhdr(x, y, text):
        ax.text(x, y, text, transform=t, ha="left", va="top",
                fontsize=_FS["tbl_sub"], color=_C["grey"], style="italic")

    def _sep(x0, x1, y):
        ax.plot([x0, x1], [y, y], transform=t,
                color=_RULE, lw=0.5, ls=":", clip_on=False)

    def _pct(v):
        return f"{v:.1f}%" if not np.isnan(v) else "—"

    def _bias(v):
        if np.isnan(v):
            return "—"
        return f"{'+'if v >= 0 else ''}{v:.2f}/bbl"

    s_sign = "+" if spread >= 0 else ""

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — sections [1]–[4]
    # ax, t, STEP are reassigned per section; helpers close over those names.
    # ══════════════════════════════════════════════════════════════════════════

    SLX = 0.030          # left text start  (card transAxes)
    SRX = 0.970          # value right-align (card transAxes)
    SHW = SRX - SLX      # heading band width passed to _hdr


    # ── [1] Run Configuration ─────────────────────────────────────────────────
    ax, t, STEP = ax_s1L, ax_s1L.transAxes, 0.075
    y = 0.940

    _hdr(SLX, y, "RUN CONFIGURATION", SHW)
    y -= 2.1 * STEP

    pr_str   = (f"±${cal_info['price_range']:.1f}/bbl"
                if "price_range" in cal_info else "N/A (Live mode)")
    el_start = selector._eligible[0].date()
    el_end   = selector._eligible[-1].date()

    _kv(SLX, SRX, y, "Simulation mode",        cal_info["mode"].capitalize());           y -= STEP
    _kv(SLX, SRX, y, "WTI Houston FOB input",  f"${wti_level:.2f}/bbl");                y -= STEP
    _kv(SLX, SRX, y, "Brent–WTI spread input", f"{s_sign}${abs(spread):.2f}/bbl");      y -= STEP
    _kv(SLX, SRX, y, "Implied Brent input",    f"${brent_level:.2f}/bbl");              y -= STEP
    _kv(SLX, SRX, y, "Σ calibration mode",     cal_info["mode"].capitalize());          y -= STEP
    _kv(SLX, SRX, y, "Σ calibration obs.",     f"{cal_info['n_sigma_obs']:,}");         y -= STEP
    _kv(SLX, SRX, y, "Pricing range used",     pr_str);                                 y -= STEP
    _kv(SLX, SRX, y, "Simulation count",       f"{n_retained:,}  (target {n_sims_target:,})"); y -= STEP
    _kv(SLX, SRX, y, "Matrix end date",        str(matrix.index[-1].date()));           y -= STEP
    _kv(SLX, SRX, y, "Eligible date range",
        f"{el_start}  →  {el_end}");                                                    y -= STEP

    # ── Mode Provenance (ax_s1R — own card with natural border) ──────────────
    ax, t = ax_s1R, ax_s1R.transAxes   # STEP stays 0.075 from section [1]
    _MP_C   = [0.100, 0.225, 0.400, 0.970]   # right-align x in ax_s1R coords
    _MP_PER = 0.425                           # Dominant Periods: left-align start
    _MAX_MP = 8

    mp_y = 0.940
    _hdr(SLX, mp_y, "MODE PROVENANCE", SHW)
    mp_y -= 2.5 * STEP
    _thdr([*_MP_C[:3], 0.970], ["Mode", "Share %", "Unique t1", ""],
          mp_y, x0=SLX)
    ax.text(_MP_PER, mp_y, "Dominant Periods", transform=t, ha="left", va="top",
            fontsize=_FS["tbl_hdr"], fontweight="bold", color=_C["dark"])
    mp_y -= STEP * 1.3

    for _r in _mp_rows[:_MAX_MP]:
        ax.text(SLX + 0.008, mp_y, _r["label"],
                transform=t, ha="left", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(_MP_C[1], mp_y, _r["share"],
                transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], fontweight="bold", color=_C["dark"])
        ax.text(_MP_C[2], mp_y, str(_r["uniq_t1"]),
                transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], fontweight="bold", color=_C["dark"])
        ax.text(_MP_PER, mp_y, _r["periods"],
                transform=t, ha="left", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        mp_y -= STEP
    if not _mp_rows:
        _subhdr(SLX + 0.008, mp_y, "No distinct modes detected")

    # ── [2] Candidate Admissibility Summary ───────────────────────────────────
    ax, t, STEP = ax_s2, ax_s2.transAxes, 0.091
    y = 0.940

    dm_max    = float(dm_all[mask_adm].max()) if mask_adm.any() else float("nan")
    date_min  = diag.get("date_min", "—")
    date_max  = diag.get("date_max", "—")
    insuf_col = _C["red"] if insufficient else _C["green"]
    insuf_str = "YES  ▲  WARNING" if insufficient else "No"

    _hdr(SLX, y, "CANDIDATE ADMISSIBILITY SUMMARY", SHW)
    y -= 2.1 * STEP

    _kv(SLX, SRX, y, "hM bandwidth",
        f"{diag['hm']:.4f}  Mahalanobis units");                                       y -= STEP
    _kv(SLX, SRX, y, "Hard filter θ",
        f"{selector._THETA:.0%}  relative deviation");                                 y -= STEP
    _kv(SLX, SRX, y, "Total candidates",       f"{n_eligible:,}");                    y -= STEP
    _kv(SLX, SRX, y, "Admissible candidates",  f"{n_admissible:,}");                  y -= STEP
    _kv(SLX, SRX, y, "Admissible %",           f"{pct_adm:.1f}%");                    y -= STEP
    _kv(SLX, SRX, y, "Insufficient candidates flag",
        insuf_str, vc=insuf_col);                                                       y -= STEP
    _kv(SLX, SRX, y, "Max D_M in admissible set",
        f"{dm_max:.4f}" if not np.isnan(dm_max) else "—");                             y -= STEP
    _kv(SLX, SRX, y, "Date span (admissible)",
        f"{date_min}  →  {date_max}");                                                 y -= STEP

    # ── [3] Input-vs-Admissible Market Check ──────────────────────────────────
    ax, t, STEP = ax_s3, ax_s3.transAxes, 0.060
    y = 0.940
    # Column right-align anchors in card transAxes: row-label | WTI | Brent
    C3 = [0.28, 0.62, 0.97]

    _hdr(SLX, y, "INPUT-VS-ADMISSIBLE MARKET CHECK", SHW)
    y -= 2.1 * STEP

    _subhdr(SLX, y, "WTI / Brent distribution — admissible candidates (unweighted percentiles)")
    y -= 1.3 * STEP

    _thdr(C3, ["", "WTI", "Brent"], y, x0=SLX)
    y -= 1.2 * STEP

    def _pc(arr, q):
        return f"${np.percentile(arr, q):.2f}" if len(arr) else "—"

    adm_rows = [
        ("Input", f"${wti_level:.2f}",                        f"${brent_level:.2f}",                  True),
        ("P5",    _pc(wti_adm,  5),                           _pc(brent_adm,  5),                     False),
        ("P25",   _pc(wti_adm, 25),                           _pc(brent_adm, 25),                     False),
        ("P50",   _pc(wti_adm, 50),                           _pc(brent_adm, 50),                     False),
        ("P75",   _pc(wti_adm, 75),                           _pc(brent_adm, 75),                     False),
        ("P95",   _pc(wti_adm, 95),                           _pc(brent_adm, 95),                     False),
        ("Mean",  f"${wti_adm.mean():.2f}"   if len(wti_adm)   else "—",
                  f"${brent_adm.mean():.2f}" if len(brent_adm) else "—",               False),
    ]
    for lbl, v_w, v_b, bold in adm_rows:
        ax.text(SLX + 0.008, y, lbl,  transform=t, ha="left", va="top",
                fontsize=_FS["tbl_body"], fontweight="bold" if bold else "normal", color=_C["dark"])
        ax.text(C3[1], y, v_w,  transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(C3[2], y, v_b,  transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        y -= STEP

    y -= 0.4 * STEP
    _sep(SLX, SRX, y)
    y -= 0.6 * STEP

    wm_wti_str   = f"${wti_wgt_mean:.2f}/bbl"   if not np.isnan(wti_wgt_mean)   else "—"
    wm_brent_str = f"${brent_wgt_mean:.2f}/bbl"  if not np.isnan(brent_wgt_mean) else "—"
    bias_str     = (f"WTI {_bias(bias_wti)}  /  Brent {_bias(bias_brent)}"
                    if not (np.isnan(bias_wti) or np.isnan(bias_brent)) else "—")

    _kv(SLX, SRX, y, "Weighted mean WTI",          wm_wti_str);   y -= STEP
    _kv(SLX, SRX, y, "Weighted mean Brent",         wm_brent_str); y -= STEP
    _kv(SLX, SRX, y, "Weighted mean bias vs input", bias_str);     y -= STEP

    # ── [4] Similarity Weight Diagnostics ─────────────────────────────────────
    ax, t, STEP = ax_s4, ax_s4.transAxes, 0.075
    y = 0.940
    # Column right-align anchors in card transAxes: band | n | % | WTI mean | Brent mean
    C4 = [0.50, 0.62, 0.72, 0.85, 0.97]

    _hdr(SLX, y, "SIMILARITY WEIGHT DIAGNOSTICS", SHW)
    y -= 2.1 * STEP

    _subhdr(SLX, y, "wi distribution — all eligible candidates")
    y -= 1.3 * STEP

    _thdr(C4, ["Band", "N", "%", "WTI mean", "Brent mean"], y, x0=SLX)
    y -= 1.2 * STEP

    bands = [
        (0.70, 1.01, "> 0.70"),
        (0.30, 0.70, "0.30 < wi <= 0.70"),
        (0.00, 0.30, "0.00 < wi <= 0.30"),
        (-1.0, 0.00, "= 0.00"),
    ]
    for lo, hi, lbl in bands:
        bm   = (wi == 0.0) if lo < 0 else ((wi > lo) & (wi <= hi))
        nb   = int(bm.sum())
        pctb = nb / n_eligible * 100 if n_eligible else 0.0
        wmu  = f"${wti_all[bm].mean():.2f}"   if nb else "—"
        bmu  = f"${brent_all[bm].mean():.2f}" if nb else "—"
        _trow(C4, [lbl, f"{nb:,}", _pct(pctb), wmu, bmu], y)
        y -= STEP

    y -= 0.4 * STEP
    _sep(SLX, SRX, y)
    y -= 0.6 * STEP

    wm_mean = wm_adm.mean() if len(wm_adm) else float("nan")
    ws_mean = ws_adm.mean() if len(ws_adm) else float("nan")
    wm_min  = wm_adm.min()  if len(wm_adm) else float("nan")

    _kv(SLX, SRX, y, "Mean w_market (admissible)",
        f"{wm_mean:.4f}" if not np.isnan(wm_mean) else "—"); y -= STEP
    _kv(SLX, SRX, y, "Mean w_season (admissible)",
        f"{ws_mean:.4f}" if not np.isnan(ws_mean) else "—"); y -= STEP
    _kv(SLX, SRX, y, "Minimum admissible w_market",
        f"{wm_min:.4f}"  if not np.isnan(wm_min)  else "—"); y -= STEP

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — sections [5]–[9]
    # ax, t, STEP are reassigned per section; helpers close over those names.
    # ══════════════════════════════════════════════════════════════════════════

    # ── [5] Top Weighted Candidates ───────────────────────────────────────────
    ax, t, STEP = ax_s5, ax_s5.transAxes, 0.070
    y = 0.940

    # Column right-align anchors (card transAxes):
    # rank | date | WTI | Brent | wi | selected_n | retained_n | aborted_n | abort_rate
    C5 = [0.100, 0.260, 0.370, 0.480, 0.570, 0.670, 0.770, 0.860, 0.970]

    _hdr(SLX, y, "TOP WEIGHTED CANDIDATES", SHW)
    y -= 2.1 * STEP

    _thdr(C5,
          ["#", "Date", "WTI", "Brent", "Wi",
           "Selected n", "Retained n", "Aborted n", "Abort rate"],
          y, x0=SLX)
    y -= 1.2 * STEP

    for rank, idx_c in enumerate(top10_wi_idx, 1):
        d_c      = wd["dates"][idx_c]
        wti_c    = wd["wti"][idx_c]
        brent_c  = wd["brent"][idx_c]
        wi_c     = wi[idx_c]
        ret_n    = retained_ctr.get(d_c, 0)
        abt_n    = overrun_ctr.get(d_c, 0)
        sel_n    = ret_n + abt_n
        abt_rate = abt_n / sel_n * 100 if sel_n else 0.0
        _trow(C5, [
            str(rank),
            str(d_c.date()),
            f"${wti_c:.2f}",
            f"${brent_c:.2f}",
            f"{wi_c:.4f}",
            f"{sel_n:,}",
            f"{ret_n:,}",
            f"{abt_n:,}",
            f"{abt_rate:.1f}%",
        ], y)
        y -= STEP

    # ── [6] Realised t1 Sampling Diagnostics ─────────────────────────────────
    ax, t, STEP = ax_s6, ax_s6.transAxes, 0.068
    y = 0.940

    # Column right-align anchors (card transAxes)
    C6a = [0.382, 0.676, 0.970]                    # row-label | WTI | Brent
    C6b = [0.350, 0.500, 0.650, 0.800, 0.970]      # date | WTI | Brent | drawn_n | draw_share

    _hdr(SLX, y, "REALISED t1 SAMPLING DIAGNOSTICS", SHW)
    y -= 2.1 * STEP

    # KV stats sit in the blank left margin (x < C6a[0]=0.382) of the t1 table.
    _KV_VAL = 0.320
    kv_pairs = [
        ("Unique t1 dates",   f"{n_unique_t1:,}"),
        ("Top 5  draw share", _pct(top5_share)),
        ("Top 10 draw share", _pct(top10_share)),
        ("Eff. n drawn t1",   f"{eff_n:.1f}" if not np.isnan(eff_n) else "—"),
        ("HHI concentration", f"{hhi:.4f}"),
    ]
    ax.text(SLX, y, "Stat.", transform=t, ha="left", va="top",
            fontsize=_FS["tbl_hdr"], fontweight="bold", color=_C["dark"])
    _thdr([_KV_VAL] + C6a, ["Value", "T1 draw", "WTI", "Brent"], y, x0=SLX, first_left=False)
    y -= 1.2 * STEP

    t1_rows = [
        ("Input", f"${wti_level:.2f}",                    f"${brent_level:.2f}",                    True),
        ("P5",    f"${np.percentile(wti_at_t1,  5):.2f}", f"${np.percentile(brent_at_t1,  5):.2f}", False),
        ("P25",   f"${np.percentile(wti_at_t1, 25):.2f}", f"${np.percentile(brent_at_t1, 25):.2f}", False),
        ("P50",   f"${np.percentile(wti_at_t1, 50):.2f}", f"${np.percentile(brent_at_t1, 50):.2f}", False),
        ("P75",   f"${np.percentile(wti_at_t1, 75):.2f}", f"${np.percentile(brent_at_t1, 75):.2f}", False),
        ("P95",   f"${np.percentile(wti_at_t1, 95):.2f}", f"${np.percentile(brent_at_t1, 95):.2f}", False),
        ("Mean",  f"${wti_at_t1.mean():.2f}",              f"${brent_at_t1.mean():.2f}",              False),
    ]
    y_kv_top = y
    for row_idx, (lbl, v_w, v_b, bold) in enumerate(t1_rows):
        if row_idx < len(kv_pairs):
            kv_lbl, kv_val = kv_pairs[row_idx]
            ax.text(SLX,     y, kv_lbl, transform=t, ha="left",  va="top",
                    fontsize=_FS["tbl_body"], color=_C["dark"])
            ax.text(_KV_VAL, y, kv_val, transform=t, ha="right", va="top",
                    fontsize=_FS["tbl_body"], fontweight="bold", color=_C["dark"])
        ax.text(C6a[0], y, lbl, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], fontweight="bold" if bold else "normal", color=_C["dark"])
        ax.text(C6a[1], y, v_w, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(C6a[2], y, v_b, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        y -= STEP
    ax.plot([_KV_VAL + 0.015, _KV_VAL + 0.015],
            [y_kv_top, y_kv_top - len(kv_pairs) * STEP],
            transform=t, color=_CARD_BORDER, lw=0.5, clip_on=False)

    _t1_wti_mean   = float(wti_at_t1.mean())
    _t1_brent_mean = float(brent_at_t1.mean())
    _kv(SLX, SRX, y, "Mean realised t1 WTI",
        f"${_t1_wti_mean:.2f}/bbl");                                              y -= STEP
    _kv(SLX, SRX, y, "Mean realised t1 Brent",
        f"${_t1_brent_mean:.2f}/bbl");                                            y -= STEP
    _kv(SLX, SRX, y, "t1 mean bias vs input",
        f"WTI {_bias(_t1_wti_mean - wti_level)}  /  "
        f"Brent {_bias(_t1_brent_mean - brent_level)}");                          y -= STEP


    # ── [7] Path Completion / Abort Diagnostics ───────────────────────────────
    ax, t, STEP = ax_s7, ax_s7.transAxes, 0.075
    y = 0.940

    # Column right-align anchors (card transAxes): row-label | WTI | Brent | n
    C7a = [0.374, 0.582, 0.782, 0.970]

    _hdr(SLX, y, "PATH COMPLETION / ABORT DIAGNOSTICS", SHW)
    y -= 2.1 * STEP

    ab_col = _C["amber"] if abort_rate > 10 else _C["dark"]
    _kv(SLX, SRX, y, "Attempted paths",         f"{n_attempted:,}");                  y -= STEP
    _kv(SLX, SRX, y, "Retained complete paths", f"{n_retained:,}",  vc=_C["green"]); y -= STEP
    _kv(SLX, SRX, y, "Aborted paths",
        f"{n_aborted:,}", vc=_C["red"] if n_aborted > 0 else _C["dark"]);             y -= STEP
    _kv(SLX, SRX, y, "Overall abort rate",      _pct(abort_rate),   vc=ab_col);      y -= STEP

    y -= 0.4 * STEP
    _sep(SLX, SRX, y)
    y -= 0.6 * STEP

    _subhdr(SLX, y, "Raw t1 vs retained t1 WTI / Brent  (mean at drawn t1 date)")
    y -= 1.3 * STEP

    _thdr(C7a, ["", "WTI", "Brent", "N"], y, x0=SLX)
    y -= 1.2 * STEP

    wti_all_drawn   = (np.concatenate([wti_at_t1, wti_ovr])     if len(wti_ovr)   else wti_at_t1)
    brent_all_drawn = (np.concatenate([brent_at_t1, brent_ovr]) if len(brent_ovr) else brent_at_t1)

    abort_rows = [
        ("Retained",  f"${wti_at_t1.mean():.2f}",       f"${brent_at_t1.mean():.2f}",       f"{n_retained:,}"),
        ("Aborted",   f"${wti_ovr.mean():.2f}"   if len(wti_ovr)   else "—",
                      f"${brent_ovr.mean():.2f}"  if len(brent_ovr) else "—",
                      f"{n_aborted:,}"),
        ("All drawn", f"${wti_all_drawn.mean():.2f}",    f"${brent_all_drawn.mean():.2f}",    f"{n_attempted:,}"),
    ]
    for lbl, v_w, v_b, v_n in abort_rows:
        ax.text(SLX + 0.008, y, lbl, transform=t, ha="left", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(C7a[1], y, v_w, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(C7a[2], y, v_b, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        ax.text(C7a[3], y, v_n, transform=t, ha="right", va="top",
                fontsize=_FS["tbl_body"], color=_C["dark"])
        y -= STEP

    # ── [9] Statistical Reliability Diagnostics ───────────────────────────────
    ax, t, STEP = ax_s9, ax_s9.transAxes, 0.060
    y = 0.940

    # Column right-align anchors: Metric | Actual | B. Mean | Bias | Std. dev.
    C9 = [0.280, 0.405, 0.525, 0.635, 0.740, 0.850, 0.970]

    ess_str      = f"{ess:.1f}"      if ess      is not None else "—"
    tail_ess_str = f"{tail_ess:.1f}" if tail_ess is not None else "—"
    ess_col      = _C["dark"]
    t_ess_col    = _C["dark"]

    _hdr(SLX, y, "STATISTICAL RELIABILITY DIAGNOSTICS", SHW)
    y -= 2.1 * STEP

    _kv(SLX, SRX, y, "Similarity set ESS", ess_str,      vc=ess_col);   y -= STEP
    _kv(SLX, SRX, y, "Tail-ESS",           tail_ess_str, vc=t_ess_col); y -= STEP

    y -= 0.4 * STEP
    _sep(SLX, SRX, y)
    y -= 0.6 * STEP

    _subhdr(SLX, y, "Bootstrap stability  (200 resamples of retained sim_df)")
    y -= 1.3 * STEP

    _thdr(C9, ["Metric", "Actual", "B. Mean", "Bias", "Std. dev.", "Min", "Max"], y, x0=SLX)
    y -= 1.2 * STEP

    def _fm(v):  return f"${v:,.0f}"
    def _fr(v):  return f"{v:.3f}"
    def _fp(v):  return f"{v:.1f}%"
    def _fe(v):  return f"{v:.1f}"
    def _fi(v):  return f"{v:.0f}"

    def _brow(lbl, actual, arr, fmt_fn, y_pos, constant=False):
        valid    = arr[~np.isnan(arr)]
        act_s    = fmt_fn(actual) if actual is not None and not np.isnan(float(actual)) else "—"
        if constant or len(valid) == 0:
            mean_s = fmt_fn(valid[0]) if len(valid) else "—"
            _trow(C9, [lbl, act_s, mean_s, "—", "—", "—", "—"], y_pos)
        else:
            mean_v  = valid.mean()
            bias_v  = actual - mean_v if actual is not None and not np.isnan(float(actual)) else float("nan")
            bias_s  = fmt_fn(bias_v) if not np.isnan(bias_v) else "—"
            _trow(C9, [lbl, act_s, fmt_fn(mean_v), bias_s,
                       fmt_fn(valid.std()), fmt_fn(valid.min()), fmt_fn(valid.max())], y_pos)

    ess_arr    = np.array([ess])      if ess      is not None else np.array([float("nan")])
    _ess_act   = ess      if ess      is not None else float("nan")
    _tess_act  = tail_ess if tail_ess is not None else float("nan")

    boot_specs = [
        ("EV ($)",          ev,          boot["ev"],        _fm,  False),
        ("VaR 0.05 ($)",    var_05,      boot["var"],       _fm,  False),
        ("|CVaR| ($)",      cvar,        boot["cvar"],      _fm,  False),
        ("EV / CVaR",       ratio,       boot["ratio"],     _fr,  False),
        ("P(loss)",         prob_loss,   boot["ploss"],     _fp,  False),
        ("ESS",             _ess_act,    ess_arr,           _fe,  True),
        ("Tail-ESS",        _tess_act,   boot["tail_ess"],  _fe,  False),
        ("Unique t1 dates", n_unique_t1, boot["unique_t1"], _fi,  False),
    ]
    for lbl, actual, arr, fmt_fn, const in boot_specs:
        _brow(lbl, actual, arr, fmt_fn, y, constant=const)
        y -= STEP

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.015,
        "Bootstrap resamples sim_df with replacement (n=200 iterations).  "
        "ESS is a pre-simulation property of the similarity weighting scheme "
        "and does not vary across resamples — Std / Min / Max are not applicable.",
        ha="center", va="bottom",
        fontsize=_FS["footnote"], color=_C["dimgrey"], style="italic",
    )

    return fig


def plot_audit_diagnostics(
    sim_df: pd.DataFrame,
    selector,
    matrix: pd.DataFrame,
    wti_level: float,
    spread: float,
    overrun_dates: list,
    cal_info: dict,
    ess: float | None,
    tail_ess: float | None,
    n_sims_target: int,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Audit Diagnostics — page 3 of the voyage P&L report.

    Produces a structured dashboard covering:
      [1] Run configuration
      [2] Candidate admissibility summary
      [3] Input-vs-admissible market check
      [4] Similarity weight diagnostics
      [5] Top 10 weighted candidates
      [6] Realised t1 sampling diagnostics
      [7] Path completion / abort diagnostics + headline P&L
      [8] Statistical reliability (ESS, Tail-ESS, bootstrap stability table)
    """
    fig = _build_audit_figure(
        sim_df=sim_df,
        selector=selector,
        matrix=matrix,
        wti_level=wti_level,
        spread=spread,
        overrun_dates=overrun_dates,
        cal_info=cal_info,
        ess=ess,
        tail_ess=tail_ess,
        n_sims_target=n_sims_target,
    )

    if save_path:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(save_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")

    if show:
        plt.show()

    return fig
