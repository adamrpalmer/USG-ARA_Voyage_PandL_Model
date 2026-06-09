"""
Freight-timing figure: σ(Δ), skew(Δ), and mean(Δ) heatmaps.

Reads  sweeps/sweep_hires_raw.csv
Writes sweeps/freight_timing.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.report import _C

_CSV     = Path("sweeps/sweep_hires_raw.csv")
_PNG_OUT = Path("sweeps/freight_timing.png")

_WTI_LEVELS    = [round(60.0 + 0.25 * i, 2) for i in range(101)]
_SPREAD_LEVELS = [round(3.50 + 0.25 * i, 2) for i in range(13)]
_DW = 0.25
_DS = 0.25

_EXTENT = [
    _WTI_LEVELS[0]    - _DW / 2,
    _WTI_LEVELS[-1]   + _DW / 2,
    _SPREAD_LEVELS[0] - _DS / 2,
    _SPREAD_LEVELS[-1] + _DS / 2,
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _nearest_idx(levels: list[float], val: float) -> int:
    return int(np.argmin(np.abs(np.asarray(levels) - val)))


def _build_grid(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return a (n_spread × n_wti) array; NaN where data is absent."""
    grid = np.full((len(_SPREAD_LEVELS), len(_WTI_LEVELS)), np.nan)
    for _, row in df.iterrows():
        ri = _nearest_idx(_SPREAD_LEVELS, float(row["spread"]))
        ci = _nearest_idx(_WTI_LEVELS,    float(row["wti"]))
        v  = float(row[col])
        grid[ri, ci] = v if np.isfinite(v) else np.nan
    return grid


def _style_ax(ax: plt.Axes) -> None:
    ax.set_xlim(_EXTENT[0], _EXTENT[1])
    ax.set_ylim(_EXTENT[2], _EXTENT[3])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.set_xlabel("WTI  ($/bbl)", fontsize=9, color=_C["dark"])
    ax.set_ylabel("Spread  ($/bbl)", fontsize=9, color=_C["dark"])
    ax.tick_params(colors=_C["grey"], which="both")
    for sp in ax.spines.values():
        sp.set_edgecolor(_C["ltgrey"])


# ── main build ────────────────────────────────────────────────────────────────

def build(df: pd.DataFrame) -> None:
    sigma_grid = _build_grid(df, "std_delta_usd") / 1e6
    skew_grid  = _build_grid(df, "skew_delta")
    mean_grid  = _build_grid(df, "mean_delta_usd") / 1e6

    XX, YY = np.meshgrid(np.array(_WTI_LEVELS), np.array(_SPREAD_LEVELS))

    import matplotlib
    cmap_sig = matplotlib.colormaps["viridis"]
    cmap_sig = cmap_sig.with_extremes(bad=_C["ltgrey"])

    cmap_skew = LinearSegmentedColormap.from_list(
        "skew_div", [_C["red"], _C["bg"], _C["blue"]], N=512,
    )
    cmap_skew.set_bad(_C["ltgrey"])

    cmap_mean = matplotlib.colormaps["PuOr"]
    cmap_mean = cmap_mean.with_extremes(bad=_C["ltgrey"])

    fin_sig   = sigma_grid[np.isfinite(sigma_grid)]
    fin_skew  = skew_grid[np.isfinite(skew_grid)]
    fin_mean  = mean_grid[np.isfinite(mean_grid)]

    vmin_sig  = (float(fin_sig.min())  - 0.005) if fin_sig.size  else 0.0
    vmax_sig  = (float(fin_sig.max())  + 0.005) if fin_sig.size  else 0.5
    vmin_skew = (float(fin_skew.min()) - 0.05)  if fin_skew.size else -1.0
    vmax_skew = (float(fin_skew.max()) + 0.05)  if fin_skew.size else  1.0
    vmin_mean = min(float(fin_mean.min()) if fin_mean.size else -0.001, -0.001)
    vmax_mean = max(float(fin_mean.max()) if fin_mean.size else  0.001,  0.001)

    norm_skew = TwoSlopeNorm(vcenter=0.0, vmin=vmin_skew, vmax=vmax_skew)
    norm_mean = TwoSlopeNorm(vcenter=0.0, vmin=vmin_mean, vmax=vmax_mean)

    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(21, 6),
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    # ── left: σ(Δ) ────────────────────────────────────────────────────────────
    im_sig = ax_l.imshow(
        sigma_grid,
        cmap=cmap_sig, vmin=vmin_sig, vmax=vmax_sig,
        interpolation="bicubic", origin="lower", extent=_EXTENT, aspect="auto",
    )
    _style_ax(ax_l)
    ax_l.set_title("Fixture-timing Volatility", fontsize=10, color=_C["dark"], pad=18)
    ax_l.text(
        0.5, 1.01,
        "σ(ΔP&L) = σ(P&L_BL − P&L_Fixture)",
        transform=ax_l.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, color=_C["grey"],
    )
    cb1 = fig.colorbar(im_sig, ax=ax_l, fraction=0.04, pad=0.03)
    cb1.ax.tick_params(labelsize=7, colors=_C["grey"])
    cb1.set_label("$M", fontsize=8, color=_C["grey"])

    # ── middle: skew(Δ) ───────────────────────────────────────────────────────
    im_skew = ax_m.imshow(
        skew_grid,
        cmap=cmap_skew, norm=norm_skew,
        interpolation="bicubic", origin="lower", extent=_EXTENT, aspect="auto",
    )
    _style_ax(ax_m)
    ax_m.set_title("Fixture-timing Skew", fontsize=10, color=_C["dark"], pad=18)
    ax_m.text(
        0.5, 1.01,
        "skew(ΔP&L) = skew(P&L_BL − P&L_Fixture)",
        transform=ax_m.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, color=_C["grey"],
    )
    cb2 = fig.colorbar(im_skew, ax=ax_m, fraction=0.04, pad=0.03)
    cb2.ax.tick_params(labelsize=7, colors=_C["grey"])

    if fin_skew.size and np.any(fin_skew > 0) and np.any(fin_skew < 0):
        ax_m.contour(
            XX, YY, skew_grid,
            levels=[0.0],
            colors=[_C["dark"]],
            linewidths=[0.8],
        )

    # ── right: mean(Δ) ────────────────────────────────────────────────────────
    im_mean = ax_r.imshow(
        mean_grid,
        cmap=cmap_mean, norm=norm_mean,
        interpolation="bicubic", origin="lower", extent=_EXTENT, aspect="auto",
    )
    _style_ax(ax_r)
    ax_r.set_title("Fixture-timing Mean", fontsize=10, color=_C["dark"], pad=18)
    ax_r.text(
        0.5, 1.01,
        "mean(ΔP&L) = mean(P&L_BL − P&L_Fixture)",
        transform=ax_r.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, color=_C["grey"],
    )
    cb3 = fig.colorbar(im_mean, ax=ax_r, fraction=0.04, pad=0.03)
    cb3.ax.tick_params(labelsize=7, colors=_C["grey"])
    cb3.set_label("$M", fontsize=8, color=_C["grey"])

    if fin_mean.size and float(fin_mean.min()) < 0.0 < float(fin_mean.max()):
        cs = ax_r.contour(
            XX, YY, mean_grid,
            levels=[0.0],
            colors=[_C["dark"]],
            linewidths=[0.8],
        )
        ax_r.clabel(cs, fmt={0.0: "mean(Δ) = 0"}, fontsize=7.5, inline=True)

    fig.savefig(_PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_PNG_OUT}")


def run() -> None:
    df = pd.read_csv(_CSV)
    build(df)


if __name__ == "__main__":
    run()
