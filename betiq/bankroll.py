"""Carnet de paris : historique, P&L, ROI, CLV et suivi de bankroll.

Tenir ce carnet est ce qui separe un parieur d'un joueur : sans mesure de la
CLV (closing line value) et du ROI sur echantillon long, impossible de savoir
si un edge est reel ou si c'est de la variance.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import Bet

DEFAULT_PATH = os.path.expanduser("~/.betiq/bankroll.json")

# Fraction de l'enjeu recuperee selon le statut (handicaps asiatiques inclus).
_PAYOUT = {
    "won": 1.0,
    "half_won": 0.5,
    "void": 0.0,
    "half_lost": -0.5,
    "lost": -1.0,
}


class Bankroll:
    def __init__(self, path: str = DEFAULT_PATH, starting: float = 1000.0):
        self.path = path
        self.starting = starting
        self.bets: list[Bet] = []
        self._load()

    # -- persistance -------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.starting = float(data.get("starting", self.starting))
        self.bets = [Bet(**b) for b in data.get("bets", [])]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "starting": self.starting,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bets": [asdict(b) for b in self.bets],
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    # -- ecriture ----------------------------------------------------------

    def add(
        self,
        sport: str,
        event: str,
        market: str,
        pick: str,
        odds: float,
        stake: float,
        model_prob: float = 0.0,
        edge: float = 0.0,
        note: str = "",
    ) -> Bet:
        bet = Bet(
            id=uuid.uuid4().hex[:8],
            placed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sport=sport,
            event=event,
            market=market,
            pick=pick,
            odds=round(float(odds), 3),
            stake=round(float(stake), 2),
            model_prob=round(float(model_prob), 4),
            edge=round(float(edge), 4),
            note=note,
        )
        self.bets.append(bet)
        self.save()
        return bet

    def settle(
        self, bet_id: str, status: str, closing_odds: float | None = None
    ) -> Bet:
        if status not in _PAYOUT:
            raise ValueError(f"Statut inconnu : {status} (attendu : {', '.join(_PAYOUT)})")
        bet = self.get(bet_id)
        bet.status = status
        if closing_odds:
            bet.closing_odds = float(closing_odds)
        frac = _PAYOUT[status]
        if frac > 0:
            bet.pnl = round(bet.stake * frac * (bet.odds - 1.0), 2)
        else:
            bet.pnl = round(bet.stake * frac, 2)
        self.save()
        return bet

    def get(self, bet_id: str) -> Bet:
        for b in self.bets:
            if b.id == bet_id:
                return b
        raise KeyError(f"Pari introuvable : {bet_id}")

    # -- lecture -----------------------------------------------------------

    @property
    def settled(self) -> list[Bet]:
        return [b for b in self.bets if b.status != "pending"]

    @property
    def pending(self) -> list[Bet]:
        return [b for b in self.bets if b.status == "pending"]

    @property
    def pnl(self) -> float:
        return round(sum(b.pnl or 0.0 for b in self.settled), 2)

    @property
    def current(self) -> float:
        """Bankroll disponible : les paris en cours sont deja engages."""
        engaged = sum(b.stake for b in self.pending)
        return round(self.starting + self.pnl - engaged, 2)

    def stats(self) -> dict[str, Any]:
        settled = self.settled
        turnover = sum(b.stake for b in settled)
        wins = sum(1 for b in settled if (b.pnl or 0) > 0)
        clvs = [b.clv for b in settled if b.clv is not None]
        return {
            "paris_regles": len(settled),
            "paris_en_cours": len(self.pending),
            "mise_totale": round(turnover, 2),
            "pnl": self.pnl,
            "roi_pct": round(100 * self.pnl / turnover, 2) if turnover else 0.0,
            "taux_reussite_pct": round(100 * wins / len(settled), 1) if settled else 0.0,
            "cote_moyenne": round(
                sum(b.odds for b in settled) / len(settled), 2
            ) if settled else 0.0,
            "clv_moyenne_pct": round(sum(clvs) / len(clvs), 2) if clvs else None,
            "clv_positive_pct": round(
                100 * sum(1 for c in clvs if c > 0) / len(clvs), 1
            ) if clvs else None,
            "bankroll_depart": self.starting,
            "bankroll_actuelle": self.current,
            "drawdown_max_pct": self.max_drawdown_pct(),
        }

    def max_drawdown_pct(self) -> float:
        equity = self.starting
        peak = equity
        worst = 0.0
        for b in sorted(self.settled, key=lambda x: x.placed_at):
            equity += b.pnl or 0.0
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return round(100 * worst, 2)

    def equity_curve(self) -> list[tuple[str, float]]:
        equity = self.starting
        out = [("depart", equity)]
        for b in sorted(self.settled, key=lambda x: x.placed_at):
            equity += b.pnl or 0.0
            out.append((b.placed_at, round(equity, 2)))
        return out


def summarise(bets: Iterable[Bet]) -> str:
    rows = list(bets)
    if not rows:
        return "Aucun pari."
    lines = [f"{'ID':<9}{'Evenement':<34}{'Selection':<20}{'Cote':>6}{'Mise':>8}{'Statut':>10}{'P&L':>9}"]
    for b in rows:
        event = b.event if len(b.event) <= 32 else b.event[:31] + "…"
        pick = f"{b.market}:{b.pick}"
        pick = pick if len(pick) <= 18 else pick[:17] + "…"
        pnl = "-" if b.pnl is None else f"{b.pnl:+.2f}"
        lines.append(
            f"{b.id:<9}{event:<34}{pick:<20}{b.odds:>6.2f}{b.stake:>8.2f}{b.status:>10}{pnl:>9}"
        )
    return "\n".join(lines)
