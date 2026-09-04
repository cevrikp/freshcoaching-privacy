"""Backtest walk-forward : le seul juge de paix d'une strategie.

Principe : a chaque etape on n'entraine le modele que sur le passe, on
predit le match suivant, et on parie uniquement si l'edge depasse le seuil.
Aucune information future n'entre dans le modele (pas de look-ahead bias).

Trois familles d'indicateurs :
  * qualite probabiliste : log-loss et Brier (plus bas = mieux) ;
  * qualite economique   : ROI, yield, nombre de paris, drawdown ;
  * calibration          : proba annoncee contre frequence observee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .engine import AgentConfig, FootballAgent
from .markets.football import football_markets
from .models import Match
from .ratings.poisson import DixonColesConfig
from .value import StakingPlan, blend_distribution, devig, edge as compute_edge


@dataclass
class BacktestResult:
    bets: int = 0
    staked: float = 0.0
    pnl: float = 0.0
    wins: int = 0
    matches_scored: int = 0
    log_loss: float = 0.0
    brier: float = 0.0
    baseline_log_loss: float = 0.0
    calibration: list[dict] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    by_edge: dict[str, dict] = field(default_factory=dict)

    @property
    def roi_pct(self) -> float:
        return round(100 * self.pnl / self.staked, 2) if self.staked else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity:
            return 0.0
        peak, worst = self.equity[0], 0.0
        for e in self.equity:
            peak = max(peak, e)
            if peak > 0:
                worst = max(worst, (peak - e) / peak)
        return round(100 * worst, 2)

    def summary(self) -> dict:
        return {
            "matchs_evalues": self.matches_scored,
            "log_loss_modele": round(self.log_loss, 4),
            "log_loss_marche": round(self.baseline_log_loss, 4),
            "gain_vs_marche": round(self.baseline_log_loss - self.log_loss, 4),
            "brier_multiclasse": round(self.brier, 4),
            "paris": self.bets,
            "mise_totale": round(self.staked, 2),
            "pnl": round(self.pnl, 2),
            "roi_pct": self.roi_pct,
            "taux_reussite_pct": round(100 * self.wins / self.bets, 1) if self.bets else 0.0,
            "drawdown_max_pct": self.max_drawdown_pct,
        }


def backtest_football(
    matches: Sequence[Match],
    train_min: int = 200,
    refit_every: int = 20,
    config: AgentConfig | None = None,
    dc_config: DixonColesConfig | None = None,
    stake_flat: float | None = None,
) -> BacktestResult:
    """Walk-forward sur des matchs contenant leurs cotes de cloture.

    `stake_flat` : mise fixe (ex. 10 EUR) au lieu du Kelly fractionne — utile
    pour comparer des strategies sans que la taille de mise brouille le signal.
    """
    config = config or AgentConfig()
    plan: StakingPlan = config.staking
    data = sorted([m for m in matches if m.closing_odds], key=lambda m: m.date)
    if len(data) <= train_min:
        raise ValueError(
            f"Pas assez de matchs avec cotes de cloture "
            f"({len(data)}) pour un backtest a partir de {train_min}."
        )

    res = BacktestResult()
    buckets: dict[int, list[tuple[float, int]]] = {}
    edge_buckets: dict[str, dict] = {}
    equity = plan.bankroll  # courbe de capital, pour le drawdown
    agent: FootballAgent | None = None

    for i in range(train_min, len(data)):
        if agent is None or (i - train_min) % refit_every == 0:
            agent = FootballAgent(config=config, dc_config=dc_config)
            agent.fit(data[:i])

        m = data[i]
        if not agent.model.knows(m.home) or not agent.model.knows(m.away):
            continue

        grid = agent.model.score_matrix(m.home, m.away)
        model_probs = football_markets(grid)["1X2"]
        prices = dict(zip(("1", "X", "2"), m.closing_odds))  # type: ignore[arg-type]
        fair = devig(prices, config.devig_method)
        blended = blend_distribution(model_probs, fair, config.weight_model)

        outcome = {"H": "1", "D": "X", "A": "2"}[m.result]
        res.matches_scored += 1
        res.log_loss += -math.log(max(blended[outcome], 1e-12))
        res.baseline_log_loss += -math.log(max(fair[outcome], 1e-12))
        res.brier += sum(
            (blended[k] - (1.0 if k == outcome else 0.0)) ** 2 for k in ("1", "X", "2")
        )

        for k, p in blended.items():
            buckets.setdefault(int(p * 10), []).append((p, 1 if k == outcome else 0))

        for pick, price in prices.items():
            p = blended[pick]
            e = compute_edge(p, price)
            if stake_flat is not None:
                stake = stake_flat if (e >= plan.min_edge and plan.min_odds <= price <= plan.max_odds) else 0.0
            else:
                _, stake = plan.stake(p, price)
            if stake <= 0:
                continue
            won = pick == outcome
            pnl = stake * (price - 1.0) if won else -stake
            res.bets += 1
            res.staked += stake
            res.pnl += pnl
            res.wins += int(won)
            equity += pnl
            res.equity.append(equity)

            key = _edge_bucket(e)
            b = edge_buckets.setdefault(key, {"paris": 0, "mise": 0.0, "pnl": 0.0})
            b["paris"] += 1
            b["mise"] += stake
            b["pnl"] += pnl

    if res.matches_scored:
        res.log_loss /= res.matches_scored
        res.baseline_log_loss /= res.matches_scored
        res.brier /= res.matches_scored

    res.calibration = [
        {
            "tranche": f"{d * 10}-{d * 10 + 10}%",
            "proba_moyenne": round(sum(p for p, _ in rows) / len(rows), 3),
            "frequence_reelle": round(sum(o for _, o in rows) / len(rows), 3),
            "n": len(rows),
        }
        for d, rows in sorted(buckets.items())
        if len(rows) >= 10
    ]
    res.by_edge = {
        k: {
            "paris": v["paris"],
            "mise": round(v["mise"], 2),
            "pnl": round(v["pnl"], 2),
            "roi_pct": round(100 * v["pnl"] / v["mise"], 2) if v["mise"] else 0.0,
        }
        for k, v in sorted(edge_buckets.items())
    }
    return res


def _edge_bucket(e: float) -> str:
    if e < 0.05:
        return "edge 3-5%"
    if e < 0.10:
        return "edge 5-10%"
    if e < 0.20:
        return "edge 10-20%"
    return "edge 20%+"
