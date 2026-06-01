"""
Fixed parameters and distribution specifications for the FOB USG-NWE Voyage P&L Model.

All monetary values in USD unless noted. All rates as decimals unless noted.
Triangular distributions are parameterised as (min, max, mode).
"""

# ── Fixed Variables (§3.4.1) ──────────────────────────────────────────────────

Q_BL_BBL        = 730_000       # cargo quantity at Bill of Lading, barrels
Q_CARGO_MT      = 95_000        # cargo quantity, metric tonnes (freight basis)

LAYTIME_DELAY_HRS    = 6        # NOR → laytime clock start delay, hours (WIBON)
LAYTIME_ALLOWANCE_HRS = 36      # laytime allowance per port, hours

DEMURRAGE_RATE_DAY = 75_000     # $/day
DEMURRAGE_RATE_HR  = DEMURRAGE_RATE_DAY / 24   # 3125 $/hr

INSURED_VALUE_MULTIPLIER = 1.10   # 110% of cargo value (Incoterms 2020 CIF)
INSURANCE_PREMIUM_RATE   = 0.005  # 0.50%

PORT_FEE_BASE_EUR = 84_000      # seaport due proxy, EUR (Port of Rotterdam tariff)

# ── Triangular Distribution Parameters (§3.3, §3.4.3) ────────────────────────
# Format: (min, max, mode)

EPS1_BPS             = (75,   200,    125)    # credit spread over SOFR, basis points
EPS2                 = (-0.002, 0.00467, 0.001) # cargo quantity loss fraction
EPS3                 = (0.2,  1.2,    0.8)    # origin port fee multiplier
EPS4                 = (0.2,  1.2,    0.8)    # destination port fee multiplier
T_SCHEDULING_LAG     = (3,    15,     5)      # days, arbitrage decision → origin port arrival
T_SETTLEMENT_LAG     = (5,    30,     10)     # days, discharge pricing → ARA settlement

# ── Decision Rule (§2) ────────────────────────────────────────────────────────

CVAR_ALPHA          = 0.05   # expected shortfall confidence level (left tail)
DECISION_THRESHOLD  = 1.0    # EV / CVaR_0.05 must meet or exceed this

# ── Simulation ────────────────────────────────────────────────────────────────

DEFAULT_N_SIMS  = 10_000

# ── Market Data Matrix Column Names (§3.5.1) ──────────────────────────────────

COL_DATED_BRENT   = "dated_brent"              # $/bbl, financial
COL_WTI_HOUSTON   = "wti_houston_fob"          # $/bbl, financial
COL_T_SEA_PASSAGE = "t_sea_passage"            # days, vessel tracking
COL_T_ORIGIN_BERTH = "t_origin_port_arr_berth_dep"   # hours, vessel tracking
COL_T_ORIGIN_PORT  = "t_origin_port_arr_port_dep"    # days,  vessel tracking
COL_T_DEST_BERTH   = "t_dest_port_arr_berth_dep"     # hours, vessel tracking
COL_WS_QUOTE      = "ws_quote"                 # WS points, financial
COL_TD25_FLAT     = "td25_flat_rate"           # $/mt, financial
COL_SOFR          = "sofr"                     # decimal rate, financial
COL_FX            = "fx_to_usd"               # USD per EUR, financial

# Columns present in the market data matrix (§3.5.1)
MATRIX_COLS = [
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
]

# Columns that follow financial-day conventions (NaN on weekends/holidays)
FINANCIAL_COLS = [COL_DATED_BRENT, COL_WTI_HOUSTON, COL_WS_QUOTE, COL_TD25_FLAT, COL_SOFR, COL_FX]

# Columns that are irregular vessel tracking observations
VESSEL_COLS = [COL_T_SEA_PASSAGE, COL_T_ORIGIN_BERTH, COL_T_ORIGIN_PORT, COL_T_DEST_BERTH]
