"""
Project AETERNUS — TradeManager (GOLD EDITION).

Vectorized GPU simulation of the Gold Standard strategy.
"""
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from config import (
    DEVICE, DTYPE, INITIAL_CAPITAL, POS_FRAC,
)
from signal_auditor import (
    apply_entry_gates,
    apply_exit_strategy,
)


@dataclass
class BatchResult:
    """Results for a batch of parameter trials."""
    net_pnl:       torch.Tensor   # [B] total PnL in USD
    win_rate:      torch.Tensor   # [B] fraction of winning trades
    n_trades:      torch.Tensor   # [B] number of trades taken
    score:         torch.Tensor   # [B] composite objective score
    avg_pnl:       torch.Tensor   # [B] average PnL per trade %


def simulate_batch(
    data: dict,
    params: torch.Tensor,
) -> BatchResult:
    """Run full trade simulation for a batch of parameter combinations.

    Args:
        data: dict of tensors from load_trade_universe()
        params: [B, N_params] parameter matrix

    Returns:
        BatchResult with metrics for each of the B trials
    """
    B = params.shape[0]
    N = data["n"]

    if N == 0:
        zeros = torch.zeros(B, dtype=DTYPE, device=DEVICE)
        return BatchResult(
            net_pnl=zeros, win_rate=zeros, n_trades=zeros,
            score=zeros, avg_pnl=zeros,
        )

    # ── Step 1: Entry gates ──
    entry_mask = apply_entry_gates(data, params)  # [B, N]

    # ── Step 2: Exit strategy (High-Fidelity) ──
    # adj_pnl_pct is the LEVERAGED pnl %
    adj_pnl_pct = apply_exit_strategy(data, params, entry_mask)  # [B, N]

    # ── Step 3: Position sizing & USD PnL ──
    base_pos_usd = INITIAL_CAPITAL * POS_FRAC
    pnl_usd = base_pos_usd * (adj_pnl_pct / 100.0)  # [B, N]

    # ── Step 4: Compute metrics ──
    n_trades = entry_mask.float().sum(dim=1)  # [B]
    net_pnl = pnl_usd.sum(dim=1)  # [B]

    # Win rate
    wins = ((adj_pnl_pct > 0) & entry_mask).float().sum(dim=1)
    win_rate = wins / n_trades.clamp(min=1)

    # Average PnL per trade (%)
    # Need to sum adj_pnl_pct for entry trades only
    avg_pnl = adj_pnl_pct.sum(dim=1) / n_trades.clamp(min=1)

    # Composite score: WR primary (>68%), PnL secondary
    # Score = WR * 100 + AvgPnL * 5
    # Penalize if n_trades < 10
    score = win_rate * 100 + avg_pnl * 5
    score = torch.where(n_trades >= 10, score, score - 500.0)

    return BatchResult(
        net_pnl=net_pnl,
        win_rate=win_rate,
        n_trades=n_trades,
        score=score,
        avg_pnl=avg_pnl,
    )


def extract_top_k(
    results: BatchResult,
    params: torch.Tensor,
    k: int = 20,
) -> list[dict]:
    """Extract top-K parameter combinations by composite score."""
    scores = results.score
    topk_vals, topk_idx = torch.topk(scores, min(k, scores.shape[0]))

    top_results = []
    for i, idx in enumerate(topk_idx):
        idx_i = idx.item()
        p = params[idx_i]
        top_results.append({
            "rank": i + 1,
            "score": results.score[idx_i].item(),
            "net_pnl": results.net_pnl[idx_i].item(),
            "win_rate": results.win_rate[idx_i].item(),
            "n_trades": int(results.n_trades[idx_i].item()),
            "avg_pnl_pct": results.avg_pnl[idx_i].item(),
            # Params
            "conf_min": p[0].item(),
            "pvt_r_min": p[1].item(),
            "midline_buffer": p[2].item(),
            "stddev_mult": p[3].item(),
            "trail_activation": p[4].item(),
        })

    return top_results


def format_result_table(results: list[dict], title: str = "Top Results") -> str:
    """Format top results as a printable table."""
    lines = []
    lines.append(f"\n{'='*95}")
    lines.append(f"  {title}")
    lines.append(f"{'='*95}")
    lines.append(
        f"{'#':>3} | {'Score':>8} | {'Net PnL':>10} | {'WR':>6} | "
        f"{'AvgPnL':>7} | {'Trades':>6} | {'Conf':>5} | {'PVT':>5} | "
        f"{'Buf':>4} | {'Std':>4} | {'Act':>5}"
    )
    lines.append("-" * 95)

    for r in results:
        lines.append(
            f"{r['rank']:>3} | {r['score']:>8.3f} | "
            f"${r['net_pnl']:>9,.0f} | {r['win_rate']:>5.1%} | "
            f"{r['avg_pnl_pct']:>6.2f}% | {r['n_trades']:>6} | "
            f"{r['conf_min']:>5.3f} | {r['pvt_r_min']:>5.3f} | "
            f"{r['midline_buffer']:>4.2f} | {r['stddev_mult']:>4.1f} | "
            f"{r['trail_activation']:>5.3f}"
        )

    lines.append("=" * 95)
    return "\n".join(lines)
