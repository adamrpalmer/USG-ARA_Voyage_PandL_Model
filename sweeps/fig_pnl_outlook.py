"""
P&L outlook figure: EV and P(Loss) heatmaps.

Reads  sweeps/sweep_hires_raw.csv
Writes sweeps/pnl_outlook.png
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
_PNG_OUT = Path("sweeps/pnl_outlook.png")

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
    ev_grid = _build_grid(df, "ev_usd") / 1e6
    pl_grid = _build_grid(df, "p_loss") * 100.0

    XX, YY = np.meshgrid(np.array(_WTI_LEVELS), np.array(_SPREAD_LEVELS))

    cmap_ev = LinearSegmentedColormap.from_list(
        "ev_div", [_C["red"], _C["bg"], _C["green"]], N=512,
    )
    cmap_ev.set_bad(_C["ltgrey"])

    cmap_pl = LinearSegmentedColormap.from_list(
        "ploss_div", [_C["green"], _C["bg"], _C["red"]], N=512,
    )
    cmap_pl.set_bad(_C["ltgrey"])

    ev_min = float(np.nanmin(ev_grid)) if np.any(np.isfinite(ev_grid)) else -1.0
    ev_max = float(np.nanmax(ev_grid)) if np.any(np.isfinite(ev_grid)) else  1.0
    norm_ev = TwoSlopeNorm(
        vcenter=0.0,
        vmin=min(ev_min, -0.05),
        vmax=max(ev_max,  0.05),
    )

    pl_min = float(np.nanmin(pl_grid)) if np.any(np.isfinite(pl_grid)) else 0.0
    pl_max = float(np.nanmax(pl_grid)) if np.any(np.isfinite(pl_grid)) else 100.0
    norm_pl = TwoSlopeNorm(
        vcenter=50.0,
        vmin=min(pl_min, 49.0),
        vmax=max(pl_max, 51.0),
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    # ── left: EV heatmap ──────────────────────────────────────────────────────
    im_ev = ax_l.imshow(
        ev_grid,
        cmap=cmap_ev, norm=norm_ev,
        interpolation="bicubic", origin="lower", extent=_EXTENT, aspect="auto",
    )
    _style_ax(ax_l)
    ax_l.set_title("Expected P&L", fontsize=10, color=_C["dark"], pad=6)
    cb1 = fig.colorbar(im_ev, ax=ax_l, fraction=0.04, pad=0.03)
    cb1.ax.tick_params(labelsize=7, colors=_C["grey"])
    cb1.set_label("$M", fontsize=8, color=_C["grey"])

    if np.any(np.isfinite(ev_grid)):
        cs_ev = ax_l.contour(
            XX, YY, ev_grid,
            levels=[0.0],
            colors=[_C["dark"]],
            linewidths=[0.8],
        )
        ax_l.clabel(cs_ev, fmt={0.0: "EV = 0"}, fontsize=7.5, inline=True)

    # ── right: P(Loss) heatmap ────────────────────────────────────────────────
    im_pl = ax_r.imshow(
        pl_grid,
        cmap=cmap_pl, norm=norm_pl,
        interpolation="bicubic", origin="lower", extent=_EXTENT, aspect="auto",
    )
    _style_ax(ax_r)
    ax_r.set_title("Expected Loss%", fontsize=10, color=_C["dark"], pad=6)
    cb2 = fig.colorbar(im_pl, ax=ax_r, fraction=0.04, pad=0.03)
    cb2.ax.tick_params(labelsize=7, colors=_C["grey"])
    cb2.set_label("%", fontsize=8, color=_C["grey"])

    if np.any(np.isfinite(pl_grid)):
        cs_pl = ax_r.contour(
            XX, YY, pl_grid,
            levels=[50.0],
            colors=[_C["dark"]],
            linewidths=[0.8],
        )
        ax_r.clabel(cs_pl, fmt={50.0: "P(Loss) = 50%"}, fontsize=7.5, inline=True)

    fig.savefig(_PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_PNG_OUT}")


def run() -> None:
    df = pd.read_csv(_CSV)
    build(df)


if __name__ == "__main__":
    run()
