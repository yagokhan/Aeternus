"""
Project AETERNUS — Exhaustive Tiered Configuration (ULTRA-AGGRESSIVE EDITION).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent
GOLD_CSV = DATA_DIR / "aeternus" / "blind_test_extended_today.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32

TITAN_ASSETS     = ["BTC", "ETH", "SOL", "BNB", "XRP"]
NAVIGATOR_ASSETS = ["ADA", "LINK", "DOT", "LTC", "BCH", "TRX", "FET", "RENDER", "NEAR", "AR", "GRT", "INJ", "THETA", "FIL", "ATOM", "ICP", "STX"]
VOLT_ASSETS      = ["PEPE", "TIA", "WIF", "BONK", "SUI", "SEI", "APT", "SHIB", "DOGE", "FLOKI", "OP"]

ASSET_TO_TIER = {a: 0 for a in TITAN_ASSETS}
ASSET_TO_TIER.update({a: 1 for a in NAVIGATOR_ASSETS})
ASSET_TO_TIER.update({a: 2 for a in VOLT_ASSETS})

TF_ENCODE = {"5m": 0, "30m": 1, "1h": 2, "4h": 3}

TRAIN_START = "2025-11-01"
TRAIN_END   = "2026-01-15"
VAL_START   = "2026-01-16"
VAL_END     = "2026-02-15"
BLIND_START = "2026-02-16"
BLIND_END   = "2026-03-29"

ROUND_TRIP_COST = 0.0012
INITIAL_CAPITAL = 15_000.0

# Aggressive Pos Sizing
POS_FRAC_BASE    = 0.12  # Double the baseline
CONF_WEIGHT_POW  = 2.0   # Scale pos by (conf^2)
PTP_TARGET_PCT   = 1.5   # Target for 50% exit
PTP_EXIT_FRAC    = 0.5   # Exit 50% at target

@dataclass
class TierSearchSpace:
    name: str
    conf_min:     tuple
    pvt_r_min:    tuple
    midline_buf:  tuple
    stddev_mult:  tuple
    hard_sl:      tuple
    activation:   tuple = (0.004, 0.010, 0.002)

# TITAN: Low risk, high leverage potential
TITAN_SPACE = TierSearchSpace(
    name="TITAN",
    conf_min=(0.900, 0.980, 0.005),
    pvt_r_min=(0.900, 0.960, 0.005),
    midline_buf=(0.10, 0.40, 0.02),
    stddev_mult=(0.8, 1.8, 0.1),
    hard_sl=(0.005, 0.012, 0.001)
)

# NAVIGATOR: Medium volatility
NAVIGATOR_SPACE = TierSearchSpace(
    name="NAVIGATOR",
    conf_min=(0.880, 0.960, 0.005),
    pvt_r_min=(0.850, 0.940, 0.005),
    midline_buf=(0.20, 0.60, 0.02),
    stddev_mult=(1.2, 2.5, 0.1),
    hard_sl=(0.010, 0.025, 0.001)
)

# VOLT: High volatility, wide stops
VOLT_SPACE = TierSearchSpace(
    name="VOLT",
    conf_min=(0.850, 0.950, 0.005),
    pvt_r_min=(0.800, 0.920, 0.005),
    midline_buf=(0.30, 0.80, 0.02),
    stddev_mult=(1.5, 4.0, 0.1),
    hard_sl=(0.015, 0.045, 0.002)
)

GPU_BATCH_SIZE = 16384
