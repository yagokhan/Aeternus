"""
Project AETERNUS — SignalAuditor (TIERED EDITION).
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from config import (
    DEVICE, DTYPE, GOLD_CSV, ASSET_TO_TIER, TF_ENCODE,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, BLIND_START,
    ROUND_TRIP_COST,
)

def load_trade_universe() -> dict:
    df = pd.read_csv(GOLD_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["entry_date"] = df["entry_ts"].dt.strftime("%Y-%m-%d")
    df["tier_id"] = df["asset"].map(ASSET_TO_TIER).fillna(1).astype(int)

    splits = {}
    for name, start, end in [
        ("train", TRAIN_START, TRAIN_END),
        ("val",   VAL_START,   VAL_END),
        ("blind", BLIND_START, "2026-03-29"), # Extended
    ]:
        mask = (df.entry_date >= start) & (df.entry_date <= end)
        splits[name] = _df_to_tensors(df[mask].copy())
    return splits

def _df_to_tensors(df: pd.DataFrame) -> dict:
    if len(df) == 0: return {"n": 0}
    return {
        "n": len(df),
        "confidence": torch.tensor(df["confidence"].values, dtype=DTYPE, device=DEVICE),
        "pvt_r":      torch.tensor(df["pvt_r"].values, dtype=DTYPE, device=DEVICE),
        "pnl_pct":    torch.tensor(df["pnl_pct"].values, dtype=DTYPE, device=DEVICE),
        "leverage":   torch.tensor(df["leverage"].values, dtype=DTYPE, device=DEVICE),
        "bars_held":  torch.tensor(df["bars_held"].values, dtype=DTYPE, device=DEVICE),
        "tier_id":    torch.tensor(df["tier_id"].values, dtype=torch.int32, device=DEVICE),
        "tf_id":      torch.tensor(df["best_tf"].map(TF_ENCODE).values, dtype=torch.int32, device=DEVICE),
    }

def apply_entry_gates(data: dict, params: torch.Tensor) -> torch.Tensor:
    B = params.shape[0]; N = data["n"]
    if N == 0: return torch.zeros(B, 0, dtype=torch.bool, device=DEVICE)
    conf = data["confidence"].unsqueeze(0); pvt = data["pvt_r"].unsqueeze(0)
    return (conf >= params[:, 0:1]) & (pvt >= params[:, 1:2])

def apply_exit_strategy(data: dict, params: torch.Tensor, entry_mask: torch.Tensor) -> torch.Tensor:
    B = params.shape[0]; N = data["n"]
    if N == 0: return torch.zeros(B, 0, dtype=DTYPE, device=DEVICE)
    hist_lev = data["leverage"].unsqueeze(0).clamp(min=1.0)
    raw_move = data["pnl_pct"].unsqueeze(0) / hist_lev
    bars = data["bars_held"].unsqueeze(0)
    
    # Params: [2] midline_buf, [3] stddev_mult, [4] activation, [5] hard_sl
    is_winner = raw_move > 0
    pfe = torch.where(is_winner, raw_move / 0.70, torch.clamp(raw_move + 0.003, min=0))
    trail_activated = pfe >= params[:, 4:5]
    trail_width = params[:, 2:3] * params[:, 3:4] * 0.001
    
    trail_pnl = torch.where(is_winner, torch.clamp(pfe - trail_width, min=0), torch.clamp(pfe - trail_width, min=-params[:, 5:6]))
    no_trail_pnl = torch.where(is_winner, raw_move, torch.clamp(raw_move, min=-params[:, 5:6]))
    
    raw_adjusted = torch.where(trail_activated, trail_pnl, no_trail_pnl) - ROUND_TRIP_COST
    return raw_adjusted * hist_lev * 100.0 * entry_mask.float()
