"""
t1 Selection Interface for the FOB USG-NWE Voyage P&L Model.

The historical dataset and the t1 selection algorithm are the two components
that are plugged in when data and methodology become available. This module
defines the interface that any t1 selector must satisfy.

Usage
-----
Implement a subclass of T1Selector and pass an instance to outer_loop():

    class MySelector(T1Selector):
        def select(self, matrix, wti_level, spread, rng):
            # ... similarity scoring logic here ...
            return chosen_date

    sim_df, overrun_dates = outer_loop(matrix, wti, spread,
                                       t1_selector=MySelector())
"""

from __future__ import annotations

import abc
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import COL_DATED_BRENT, COL_WTI_HOUSTON


class T1Selector(abc.ABC):
    """
    Abstract base for all t1 selection strategies.

    A selector receives the historical dataset and the live inputs at the
    arbitrage decision (t=0), and returns a single historical date t1 from
    which the inner loop will begin tracing the matrix block.

    Contract
    --------
    - The returned date must be a valid index entry in `matrix`.
    - The returned date must be early enough that the full block (up to
      MAX_BLOCK_DAYS ahead) remains within the matrix.
    - The selector may use `rng` for any stochastic choices, so that results
      are reproducible when the outer loop seeds the generator.
    """

    @abc.abstractmethod
    def select(
        self,
        matrix: pd.DataFrame,
        wti_level: float,
        spread: float,
        rng: np.random.Generator,
    ) -> pd.Timestamp:
        """
        Choose a historical block-start date t1.

        Parameters
        ----------
        matrix    : Historical dataset (DatetimeIndex, ascending).
        wti_level : Current WTI Houston FOB price ($/bbl) at t=0.
        spread    : Current Brent–WTI spread ($/bbl) at t=0.
        rng       : NumPy random generator (seeded by outer loop).

        Returns
        -------
        pd.Timestamp — a date present in matrix.index.
        """

    # ── Shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _eligible_dates(matrix: pd.DataFrame) -> pd.DatetimeIndex:
        """
        All matrix dates with both WTI and Brent non-NaN, excluding the first
        row (t0 = t1 − 1 must remain within the matrix).

        No tail cutoff is applied.  Blocks that would overrun the matrix end
        are detected and aborted during inner_loop (BlockOverrunError); the
        outer loop retries until the target simulation count is reached.
        """
        start = matrix.index[0]
        valid_mask = (
            matrix[COL_WTI_HOUSTON].notna()
            & matrix[COL_DATED_BRENT].notna()
            & (matrix.index > start)
        )
        return matrix.index[valid_mask]


# ── Mahalanobis market–seasonal similarity selector ──────────────────────────

class MahalanobisT1Selector(T1Selector):
    """
    §3.5.3 Step 2 — t1 selection via joint market–seasonal similarity scoring
    with hard market admissibility filter (eqs. 18–30).

    Two covariance calibration modes (§3.5.3):
      Live     — Σ and hM estimated from the 3 calendar months of trading
                 days immediately preceding t=0.  Best when live WTI/Brent
                 inputs match current market levels.
      Scenario — Σ and hM estimated from observations within ±$10/bbl of
                 the desired WTI and Brent levels (expanded by $2.5/bbl
                 increments until ≥ 50 observations are available).

    In both modes the calibration set C is used only to estimate Σ (eq. 19)
    and hM (eq. 22).  Candidate block-start dates are drawn from the full
    eligible set D via the combined market–seasonal weighting (eqs. 20–26).
    A hard admissibility filter (eq. 20, θ=0.05) zeros the market weight for
    any observation whose WTI or Brent deviates from the input by more than 5%
    in relative terms.  Sampling probabilities are normalised over all of D;
    if Σwᵢ = 0 the selector raises an error and does not normalise.

    calibrate() must be called once (after prices are known) before any
    call to select() or compute_ess().
    """

    _H_SEASON: float          = 60.0    # seasonal bandwidth, days (§3.5.3 eq. 24)
    _THETA: float             = 0.05    # hard admissibility relative threshold (§3.5.3 eq. 20)
    _LIVE_WINDOW_DAYS: int    = 91      # ~3 calendar months preceding t=0
    _SCENARIO_INIT_RANGE: float = 10.0  # ±$/bbl initial pricing range (§3.5.3)
    _SCENARIO_STEP: float     = 2.5     # incremental range expansion, $/bbl
    _SCENARIO_MIN_OBS: int    = 50      # minimum observations for Σ estimation

    def __init__(self, matrix: pd.DataFrame, mode: str = "live") -> None:
        """
        Cache eligible candidate arrays.  Σ and hM are deferred to calibrate().

        Parameters
        ----------
        matrix : Historical dataset (DatetimeIndex, ascending).
        mode   : 'live' or 'scenario' — governs how Σ is estimated.
        """
        mode = mode.strip().lower()
        if mode not in ("live", "scenario"):
            raise ValueError("mode must be 'live' or 'scenario'.")
        self._mode    = mode
        self._matrix  = matrix

        eligible = self._eligible_dates(matrix)
        if len(eligible) < 10:
            raise ValueError(
                f"Only {len(eligible)} eligible t1 dates — need ≥ 10 for "
                "Mahalanobis selector."
            )
        self._eligible: pd.DatetimeIndex = eligible
        self._candidates: np.ndarray = np.column_stack([
            matrix.loc[eligible, COL_WTI_HOUSTON].values,
            matrix.loc[eligible, COL_DATED_BRENT].values,
        ])  # shape (n_eligible, 2)

        # Day-of-year for each eligible date (1–365/366), used for seasonal scoring
        self._doy: np.ndarray = np.array(
            [d.timetuple().tm_yday for d in eligible], dtype=float
        )

        # Set by calibrate() before first use
        self._sigma_inv: np.ndarray | None = None
        self._hm: float | None             = None

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        wti_level: float,
        spread: float,
        reference_date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Estimate Σ (eq. 19) and hM (eq. 21) from the mode-appropriate
        calibration set C.  Must be called once per run before select() or
        compute_ess().

        Live mode     — C = trading days in the 3 months preceding t=0.
        Scenario mode — C = rows within ±$10/bbl of both WTI and Brent inputs,
                        expanded by $2.5/bbl until ≥ 50 observations are found.

        Parameters
        ----------
        wti_level      : WTI Houston FOB at t=0 ($/bbl).
        spread         : Brent–WTI spread at t=0 ($/bbl).
        reference_date : Date used as t=0 for Live windowing.  Defaults to today.

        Returns
        -------
        dict with keys:
            mode        — 'live' or 'scenario'
            n_sigma_obs — number of observations used to estimate Σ
            price_range — (scenario only) final ±$/bbl range applied
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today().normalize()

        matrix      = self._matrix
        brent_level = wti_level + spread
        both_valid  = (
            matrix[COL_WTI_HOUSTON].notna() & matrix[COL_DATED_BRENT].notna()
        )

        cal_info: dict = {"mode": self._mode}

        if self._mode == "live":
            window_start = reference_date - timedelta(days=self._LIVE_WINDOW_DAYS)
            mask = (
                both_valid
                & (matrix.index >= window_start)
                & (matrix.index <= reference_date)
            )
            cal_set = matrix.loc[mask]
            n_obs   = len(cal_set)
            if n_obs < 4:
                raise ValueError(
                    f"Live calibration window ({window_start.date()} – "
                    f"{reference_date.date()}) contains only {n_obs} valid "
                    "(WTI, Brent) observations. Ensure the matrix covers the "
                    "3-month window preceding today."
                )
            cal_info["n_sigma_obs"] = n_obs

        else:  # scenario
            price_range = self._SCENARIO_INIT_RANGE
            while True:
                mask = (
                    both_valid
                    & (matrix[COL_WTI_HOUSTON] >= wti_level    - price_range)
                    & (matrix[COL_WTI_HOUSTON] <= wti_level    + price_range)
                    & (matrix[COL_DATED_BRENT] >= brent_level  - price_range)
                    & (matrix[COL_DATED_BRENT] <= brent_level  + price_range)
                )
                n_obs = int(mask.sum())
                if n_obs >= self._SCENARIO_MIN_OBS:
                    break
                price_range += self._SCENARIO_STEP
            cal_set = matrix.loc[mask]
            cal_info["n_sigma_obs"] = n_obs
            cal_info["price_range"] = price_range

        # Eq. (19): Σ from calibration set C
        wti_cal   = cal_set[COL_WTI_HOUSTON].values
        brent_cal = cal_set[COL_DATED_BRENT].values
        cov = np.cov(np.column_stack([wti_cal, brent_cal]).T)
        self._sigma_inv = np.linalg.inv(cov)

        # Eq. (21): hM = median D_M over calibration set C
        x0       = np.array([wti_level, brent_level])
        cal_pts  = np.column_stack([wti_cal, brent_cal])
        diffs    = cal_pts - x0
        tmp      = diffs @ self._sigma_inv
        dm_cal   = np.sqrt(np.maximum(np.sum(tmp * diffs, axis=1), 0.0))
        self._hm = float(np.median(dm_cal))
        if self._hm == 0.0:
            self._hm = 1e-10

        return cal_info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mahalanobis(self, wti_level: float, spread: float) -> np.ndarray:
        """Eq. (18): D_M(xᵢ, x₀) for all eligible candidates using stored Σ."""
        if self._sigma_inv is None:
            raise RuntimeError("Call calibrate() before using the selector.")
        x0    = np.array([wti_level, wti_level + spread])
        diffs = self._candidates - x0
        tmp   = diffs @ self._sigma_inv
        return np.sqrt(np.maximum(np.sum(tmp * diffs, axis=1), 0.0))

    def _joint_weights(
        self,
        wti_level: float,
        spread: float,
        reference_doy: int,
    ) -> np.ndarray:
        """
        Compute combined market–seasonal similarity scores wᵢ for all candidates.

        Eq. (20): hard admissibility filter I_market zeros the weight for any
        observation whose WTI or Brent deviates from inputs by more than θ=5%.
        Eq. (21): w_market,i = I_market,i × exp(−D_M²/(2hM²)).
        Eq. (24)–(25): wᵢ = w_market,i × w_season,i.
        Returns raw (unnormalised) weights; normalisation is over all of D.
        """
        if self._hm is None:
            raise RuntimeError("Call calibrate() before using the selector.")

        brent_level = wti_level + spread

        # Eq. (20): hard market admissibility filter, θ = 0.05
        wti_rel_dev   = np.abs(self._candidates[:, 0] - wti_level)   / wti_level
        brent_rel_dev = np.abs(self._candidates[:, 1] - brent_level) / brent_level
        I_market = (
            (wti_rel_dev   <= self._THETA) &
            (brent_rel_dev <= self._THETA)
        ).astype(float)

        dm = self._mahalanobis(wti_level, spread)

        # Eq. (21): market similarity score, zero for inadmissible observations
        wmarket = I_market * np.exp(-(dm ** 2) / (2.0 * self._hm ** 2))

        # Eq. (23): circular day-of-year distances
        doy_diff = np.abs(self._doy - reference_doy)
        dseason  = np.minimum(doy_diff, 365.0 - doy_diff)

        # Eq. (24): seasonal similarity scores, hs = 60 days
        wseason = np.exp(-(dseason ** 2) / (2.0 * self._H_SEASON ** 2))

        # Eq. (25): combined joint similarity weight
        return wmarket * wseason

    def _normalise(self, wi: np.ndarray) -> np.ndarray:
        """
        Eq. (26): normalise joint weights over all of D into sampling probabilities.
        Raises ValueError if Σwᵢ = 0 (no admissible candidates).
        """
        total = wi.sum()
        if total == 0.0:
            raise ValueError(
                "No admissible t1 candidates: all historical observations fall "
                f"outside the ±{self._THETA:.0%} relative price window for both "
                "WTI and Brent. Check inputs or extend the dataset."
            )
        return wi / total

    # ── Public interface ──────────────────────────────────────────────────────

    def select(
        self,
        matrix: pd.DataFrame,
        wti_level: float,
        spread: float,
        rng: np.random.Generator,
    ) -> pd.Timestamp:
        """
        §3.5.3 Step 2 — draw t1 via joint market–seasonal similarity scoring.

        Eqs. (20)–(26): apply hard admissibility filter, compute joint weights,
        normalise over all of D, and draw one candidate.
        calibrate() must have been called first.
        """
        doy0 = pd.Timestamp.today().timetuple().tm_yday
        wi   = self._joint_weights(wti_level, spread, doy0)
        pi   = self._normalise(wi)
        chosen = rng.choice(len(self._eligible), p=pi)
        return self._eligible[chosen]

    def compute_ess(
        self,
        wti_level: float,
        spread: float,
        reference_date: pd.Timestamp | None = None,
    ) -> float:
        """
        §3.5.3 Eq. (27) — ESS_D over all of D.

        ESS_D = 1 / Σ(pᵢ²) where pᵢ are the normalised sampling probabilities
        over all eligible candidates D.  calibrate() must have been called first.
        Returns NaN if Σwᵢ = 0 (no admissible candidates).
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today()
        doy0 = reference_date.timetuple().tm_yday

        wi = self._joint_weights(wti_level, spread, doy0)
        if wi.sum() == 0.0:
            return float("nan")
        pi = wi / wi.sum()
        return float(1.0 / np.sum(pi ** 2))

    def n_candidates(
        self,
        wti_level: float,
        spread: float,
        reference_date: pd.Timestamp | None = None,
    ) -> int:
        """
        §3.5.3 Eq. (30) — N_candidates = #{i ∈ D : pᵢ > 0}.

        Equal to the number of observations passing the hard admissibility
        filter (I_market,i = 1), since seasonal weight is always positive.
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today()
        doy0 = reference_date.timetuple().tm_yday
        wi   = self._joint_weights(wti_level, spread, doy0)
        return int((wi > 0).sum())

    def diagnostic_admissible_set(
        self,
        wti_level: float,
        spread: float,
        reference_date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Audit the admissible candidate set (I_market,i = 1, eq. 20).

        Decomposes the joint weight into its market and seasonal components for
        every eligible candidate and reports the distribution of WTI/Brent levels
        and Mahalanobis distances for admissible observations.  The pᵢ-weighted
        mean reveals the effective centre of mass of the simulation.

        Returns
        -------
        dict with:
            hm                   — Gaussian kernel bandwidth (from calibration set)
            n_eligible           — total eligible candidates
            n_admissible         — candidates passing the hard admissibility filter
            n_candidates         — #{i ∈ D : pᵢ > 0}  (= n_admissible)
            insufficient         — True if Σwᵢ = 0 (no admissible candidates)
            wti_input            — user-supplied WTI level
            brent_input          — implied Brent level
            wti_p10/p25/p50/p75/p90 — unweighted percentiles of admissible WTI values
            wti_weighted_mean    — pᵢ-weighted mean WTI of the admissible set
            dm_max_admissible    — largest Mahalanobis distance in the admissible set
            date_min / date_max  — date span of the admissible candidates
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today()
        doy0 = reference_date.timetuple().tm_yday

        wi    = self._joint_weights(wti_level, spread, doy0)
        mask  = wi > 0   # I_market,i = 1 (seasonal weight is always > 0)
        total = wi.sum()

        insufficient  = total == 0.0
        n_admissible  = int(mask.sum())

        if insufficient or n_admissible == 0:
            return {
                "hm":               self._hm,
                "n_eligible":       int(len(self._eligible)),
                "n_admissible":     0,
                "n_candidates":     0,
                "insufficient":     True,
                "wti_input":        wti_level,
                "brent_input":      wti_level + spread,
            }

        pi_full        = wi / total
        pi_admissible  = pi_full[mask]
        wti_pass       = self._candidates[mask, 0]
        dm             = self._mahalanobis(wti_level, spread)
        dates_pass     = self._eligible[mask]

        return {
            "hm":               self._hm,
            "n_eligible":       int(len(self._eligible)),
            "n_admissible":     n_admissible,
            "n_candidates":     n_admissible,
            "insufficient":     False,
            "wti_input":        wti_level,
            "brent_input":      wti_level + spread,
            "wti_p10":          float(np.percentile(wti_pass, 10)),
            "wti_p25":          float(np.percentile(wti_pass, 25)),
            "wti_p50":          float(np.percentile(wti_pass, 50)),
            "wti_p75":          float(np.percentile(wti_pass, 75)),
            "wti_p90":          float(np.percentile(wti_pass, 90)),
            "wti_weighted_mean": float(np.dot(pi_admissible, wti_pass)),
            "dm_max_admissible": float(dm[mask].max()),
            "date_min":          str(dates_pass.min().date()),
            "date_max":          str(dates_pass.max().date()),
        }

    def weight_distribution(
        self,
        wti_level: float,
        spread: float,
        reference_date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Return raw weight arrays for every eligible candidate.

        Useful for auditing the wᵢ distribution before running simulations.

        Returns
        -------
        dict with numpy arrays:
            wi        — joint weight wᵢ = wmarket × wseason  (eq. 25)
            wmarket   — market similarity component (includes I_market, eq. 21)
            wseason   — seasonal similarity component (eq. 24)
            I_market  — hard admissibility indicator (eq. 20), 0 or 1
            dm        — Mahalanobis distance from x₀
            wti       — WTI level at each eligible date
            brent     — Brent level at each eligible date
            dates     — eligible DatetimeIndex
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today()
        doy0 = reference_date.timetuple().tm_yday

        brent_level   = wti_level + spread
        wti_rel_dev   = np.abs(self._candidates[:, 0] - wti_level)   / wti_level
        brent_rel_dev = np.abs(self._candidates[:, 1] - brent_level) / brent_level
        I_market = (
            (wti_rel_dev   <= self._THETA) &
            (brent_rel_dev <= self._THETA)
        ).astype(float)

        dm       = self._mahalanobis(wti_level, spread)
        doy_diff = np.abs(self._doy - doy0)
        dseason  = np.minimum(doy_diff, 365.0 - doy_diff)
        wmarket  = I_market * np.exp(-(dm ** 2) / (2.0 * self._hm ** 2))
        wseason  = np.exp(-(dseason ** 2) / (2.0 * self._H_SEASON ** 2))
        wi       = wmarket * wseason

        return {
            "wi":       wi,
            "wmarket":  wmarket,
            "wseason":  wseason,
            "I_market": I_market,
            "dm":       dm,
            "wti":      self._candidates[:, 0].copy(),
            "brent":    self._candidates[:, 1].copy(),
            "dates":    self._eligible,
        }

    def compute_tail_ess(
        self,
        sim_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> float:
        """
        §3.5.2 Eqs. (29–30) — Tail effective sample size.

        ESS_T = 1 / Σ(rᵢ²)

        where rᵢ = (number of simulation paths in the left-tail generated from
        dateᵢ) / (number of observations in the left-tail).

        Parameters
        ----------
        sim_df : DataFrame returned by outer_loop(); must contain 'pnl' and
                 'node1_date'.
        alpha  : Left-tail quantile cutoff (default 0.05, matching CVaR alpha).
        """
        pnl       = sim_df["pnl"]
        threshold = float(pnl.quantile(alpha))
        tail_mask = pnl <= threshold
        n_tail    = int(tail_mask.sum())
        if n_tail == 0:
            return float("nan")

        tail_dates = sim_df.loc[tail_mask, "node1_date"]
        counts     = tail_dates.value_counts()
        r          = counts.values / n_tail
        return float(1.0 / np.sum(r ** 2))
