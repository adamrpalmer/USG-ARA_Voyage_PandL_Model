"""
P&L sub-functions for the FOB USG-NWE Voyage P&L Model.

Each function maps directly to a numbered equation in the methodology document.
All monetary outputs are in USD.

Equation references (V6):
    (1)     π = Gross Cargo Margin − Freight − Financing − Demurrage − Insurance − Port fees
    (2-6)   Gross Cargo Margin (incl. Q_discharge eq. 4)
    (7)     Freight
    (8-9)   Financing
    (10-12) Demurrage
    (13)    Insurance
    (14-16) Port fees
    (17)    Scheduling lag
"""

from .config import (
    Q_BL_BBL,
    Q_CARGO_MT,
    LAYTIME_DELAY_HRS,
    LAYTIME_ALLOWANCE_HRS,
    DEMURRAGE_RATE_HR,
    INSURED_VALUE_MULTIPLIER,
    INSURANCE_PREMIUM_RATE,
    PORT_FEE_BASE_EUR,
)


def compute_cargo_at_discharge(
    eps2: float,
    q_bl_bbl: float = Q_BL_BBL,
) -> float:
    """
    Eq. (4): Q_discharge = Q_BL × (1 − ε₂)

    ε₂ ~ Tri(−0.002, 0.00467, 0.001). Positive ε₂ = cargo loss (evaporation,
    handling); negative ε₂ = apparent gain (measurement error).
    """
    return q_bl_bbl * (1.0 - eps2)


def compute_gross_cargo_margin(
    p_brent_5day: float,
    q_discharge_bbl: float,
    p_wti_5day: float,
    q_bl_bbl: float = Q_BL_BBL,
) -> float:
    """
    Eq. (2): Gross Cargo Margin = (P_Brent,discharge · Q_discharge) − (P_WTI,BL · Q_BL)

    p_brent_5day    : 5-day average Dated Brent around discharge ($/bbl)
    q_discharge_bbl : cargo quantity at discharge (bbl)
    p_wti_5day      : 5-day average WTI Houston FOB around BL ($/bbl)
    q_bl_bbl        : cargo quantity at BL (bbl)
    """
    return p_brent_5day * q_discharge_bbl - p_wti_5day * q_bl_bbl



def compute_freight(
    ws_quote: float,
    td25_flat_rate: float,
    q_cargo_mt: float = Q_CARGO_MT,
) -> float:
    """
    Eq. (7): Freight = (WS / 100) × TD25_flat × Q_cargo_mt

    ws_quote       : Worldscale quote (WS points), fixed at arbitrage decision t=0
    td25_flat_rate : TD25 flat rate ($/mt), fixed at arbitrage decision t=0
    q_cargo_mt     : cargo quantity in metric tonnes
    """
    return (ws_quote / 100.0) * td25_flat_rate * q_cargo_mt


def compute_financing(
    sofr_bl: float,
    eps1_bps: float,
    p_wti_5day: float,
    financing_exposure_days: float,
    q_bl_bbl: float = Q_BL_BBL,
) -> float:
    """
    Eq. (8): Financing = (SOFR_BL + ε₁) × (P_WTI · Q_BL) × (exposure / 365)

    sofr_bl                 : SOFR 30-day average at BL date (decimal, e.g. 0.053)
    eps1_bps                : credit spread over SOFR, ε₁ ~ Tri(75, 200, 125) bps
    p_wti_5day              : 5-day average WTI Houston FOB at BL ($/bbl)
    q_bl_bbl                : cargo quantity at BL (bbl)
    financing_exposure_days : Node 6 − Node 3 in calendar days (ACT/365)
    """
    rate      = sofr_bl + eps1_bps / 10_000.0
    principal = p_wti_5day * q_bl_bbl
    return rate * principal * (financing_exposure_days / 365.0)


def compute_demurrage(
    t_origin_berth_hrs: float,
    t_dest_berth_hrs: float,
) -> float:
    """
    Eqs. (10-12): Demurrage = D_origin + D_destination

    D_port = max(0, max(0, t_port_arr→berth_dep − 6) − 36) × r_demurrage

    t_origin_berth_hrs : hours from origin port arrival to berth departure
    t_dest_berth_hrs   : hours from destination port arrival to berth departure

    Laytime clock delay : 6 hrs post-NOR (WIBON).
    Laytime allowance   : 36 hrs per port, SHINC, non-reversible.
    Demurrage rate      : $75,000/day = $3,125/hr.
    """
    def _port_demurrage(t_hrs: float) -> float:
        billable = max(0.0, t_hrs - LAYTIME_DELAY_HRS) - LAYTIME_ALLOWANCE_HRS
        return max(0.0, billable) * DEMURRAGE_RATE_HR

    return _port_demurrage(t_origin_berth_hrs) + _port_demurrage(t_dest_berth_hrs)


def compute_insurance(
    p_wti_5day: float,
    q_bl_bbl: float = Q_BL_BBL,
) -> float:
    """
    Eq. (13): Insurance = P_WTI · Q_BL × 110% × 0.50%

    Insured value = 110% of cargo value (Incoterms 2020 CIF requirement).
    Premium       = 0.50% of insured value (Institute Cargo Clauses A).
    """
    return p_wti_5day * q_bl_bbl * INSURED_VALUE_MULTIPLIER * INSURANCE_PREMIUM_RATE


def compute_port_fees(
    fx_bl: float,
    fx_discharge: float,
    eps3: float,
    eps4: float,
    base_eur: float = PORT_FEE_BASE_EUR,
) -> float:
    """
    Eqs. (14-16): Port fees in USD  (numbering unchanged in V6)

    Origin fees      = FX_BL       × (84,000 + ε₃ × 84,000)
    Destination fees = FX_discharge × (84,000 + ε₄ × 84,000)

    fx_bl, fx_discharge : USD per EUR at BL and discharge dates respectively
    eps3, eps4          : port expense multipliers, ε₃,ε₄ ~ Tri(0.2, 1.2, 0.8)
    base_eur            : seaport due proxy (Port of Rotterdam tariff, in EUR)
    """
    origin_eur = base_eur * (1.0 + eps3)
    dest_eur   = base_eur * (1.0 + eps4)
    return fx_bl * origin_eur + fx_discharge * dest_eur


def compute_pnl(
    p_brent_5day: float,
    p_wti_5day: float,
    eps2: float,
    ws_quote: float,
    td25_flat_rate: float,
    sofr_bl: float,
    eps1_bps: float,
    financing_exposure_days: float,
    t_origin_berth_hrs: float,
    t_dest_berth_hrs: float,
    fx_bl: float,
    fx_discharge: float,
    eps3: float,
    eps4: float,
    q_bl_bbl: float = Q_BL_BBL,
) -> dict:
    """
    Eq. (1): π = Gross Cargo Margin − Freight − Financing − Demurrage − Insurance − Port fees

    Returns a dict with 'pnl' (USD) and each component. The 'spread' key holds
    the Gross Cargo Margin value (retained for downstream compatibility).
    """
    q_discharge = compute_cargo_at_discharge(eps2, q_bl_bbl)

    spread    = compute_gross_cargo_margin(p_brent_5day, q_discharge, p_wti_5day, q_bl_bbl)
    freight   = compute_freight(ws_quote, td25_flat_rate)
    financing = compute_financing(sofr_bl, eps1_bps, p_wti_5day,
                                  financing_exposure_days, q_bl_bbl)
    demurrage = compute_demurrage(t_origin_berth_hrs, t_dest_berth_hrs)
    insurance = compute_insurance(p_wti_5day, q_bl_bbl)
    port_fees = compute_port_fees(fx_bl, fx_discharge, eps3, eps4)

    pnl = spread - freight - financing - demurrage - insurance - port_fees

    return {
        "pnl":       pnl,
        "spread":    spread,
        "freight":   freight,
        "financing": financing,
        "demurrage": demurrage,
        "insurance": insurance,
        "port_fees": port_fees,
    }
