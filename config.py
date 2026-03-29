"""
Project AETERNUS — Exhaustive Tiered Configuration (GOLD EDITION).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent
GOLD_CSV = DATA_DIR / "varanus-neo-flow-hybrid-extended" / "blind_test_trades.csv"
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
POS_FRAC        = 0.06

@dataclass
class TierSearchSpace:
    name: str
    conf_min:     tuple
    pvt_r_min:    tuple
    midline_buf:  tuple
    stddev_mult:  tuple
    activation:   tuple = (0.004, 0.010, 0.002)

# TITAN: Ultra-low noise, high precision
TITAN_SPACE = TierSearchSpace(
    name="TITAN",
    conf_min=(0.920, 0.980, 0.002),
    pvt_r_min=(0.900, 0.960, 0.002),
    midline_buf=(0.15, 0.60, 0.02),
    stddev_mult=(0.8, 2.5, 0.1)
)

# NAVIGATOR: Medium volatility
NAVIGATOR_SPACE = TierSearchSpace(
    name="NAVIGATOR",
    conf_min=(0.900, 0.970, 0.002),
    pvt_r_min=(0.880, 0.950, 0.002),
    midline_buf=(0.25, 0.80, 0.02),
    stddev_mult=(1.2, 3.0, 0.1)
)

# VOLT: High volatility, wide ranges
VOLT_SPACE = TierSearchSpace(
    name="VOLT",
    conf_min=(0.880, 0.960, 0.002),
    pvt_r_min=(0.850, 0.940, 0.002),
    midline_buf=(0.35, 1.00, 0.02),
    stddev_mult=(1.5, 4.5, 0.1)
)

GPU_BATCH_SIZE = 16384
