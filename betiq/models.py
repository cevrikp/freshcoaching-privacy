"""Structures de donnees partagees par tout l'agent.

Volontairement en dataclasses stdlib : aucune dependance externe n'est
requise pour faire tourner le coeur de l'agent (modeles, marches, mises).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Iterable


# --------------------------------------------------------------------------
# Resultats historiques
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """Un match termine, utilise pour l'entrainement des modeles."""

    date: date
    home: str
    away: str
    home_score: int
    away_score: int
    league: str = ""
    # Cotes de cloture si disponibles (utile pour le backtest / la CLV).
    closing_odds: tuple[float, float, float] | None = None

    @property
    def result(self) -> str:
        if self.home_score > self.away_score:
            return "H"
        if self.home_score < self.away_score:
            return "A"
        return "D"

    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score


@dataclass(frozen=True)
class MapResult:
    """Une map / manche d'esport (pas de match nul possible)."""

    date: date
    winner: str
    loser: str
    game: str = ""
    event: str = ""
    # Score interne de la map (ex. rounds CS2 16-12), optionnel.
    winner_score: int | None = None
    loser_score: int | None = None


# --------------------------------------------------------------------------
# Rencontres a venir + cotes
# --------------------------------------------------------------------------


@dataclass
class Odds:
    """Cotes decimales d'un marche, indexees par selection."""

    market: str
    prices: dict[str, float]
    bookmaker: str = ""

    @property
    def overround(self) -> float:
        """Marge du bookmaker : somme des probabilites implicites brutes."""
        return sum(1.0 / p for p in self.prices.values() if p and p > 1.0)

    @property
    def margin_pct(self) -> float:
        return (self.overround - 1.0) * 100.0


@dataclass
class Fixture:
    """Une rencontre a venir."""

    home: str
    away: str
    kickoff: datetime | None = None
    league: str = ""
    sport: str = "football"
    # Pour l'esport : format de la serie (1 = map unique, 3 = Bo3, 5 = Bo5).
    best_of: int = 1
    odds: list[Odds] = field(default_factory=list)

    def odds_for(self, market: str) -> Odds | None:
        for o in self.odds:
            if o.market == market:
                return o
        return None

    @property
    def label(self) -> str:
        return f"{self.home} vs {self.away}"


# --------------------------------------------------------------------------
# Sorties du moteur
# --------------------------------------------------------------------------


@dataclass
class Selection:
    """Une selection evaluee : proba modele, cote, edge, mise conseillee."""

    market: str
    pick: str
    model_prob: float
    odds: float | None = None
    fair_prob: float | None = None      # proba du marche apres retrait de la marge
    blended_prob: float | None = None   # modele + marche (shrinkage)
    edge: float | None = None           # esperance par euro mise
    kelly: float | None = None          # fraction de bankroll (deja bridee)
    stake: float | None = None          # mise en euros
    confidence: str = "moyenne"
    rationale: list[str] = field(default_factory=list)

    @property
    def used_prob(self) -> float:
        return self.blended_prob if self.blended_prob is not None else self.model_prob

    @property
    def fair_odds(self) -> float:
        p = self.used_prob
        return float("inf") if p <= 0 else 1.0 / p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fair_odds"] = None if math.isinf(self.fair_odds) else round(self.fair_odds, 3)
        return d


@dataclass
class Prediction:
    """Le pronostic complet d'une rencontre."""

    fixture: Fixture
    probabilities: dict[str, dict[str, float]]  # marche -> selection -> proba
    expected: dict[str, float] = field(default_factory=dict)  # buts attendus, etc.
    selections: list[Selection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def best_bet(self) -> Selection | None:
        candidates = [s for s in self.selections if s.edge is not None and s.stake]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.edge or 0.0))


# --------------------------------------------------------------------------
# Suivi de bankroll
# --------------------------------------------------------------------------


@dataclass
class Bet:
    """Un pari enregistre dans le carnet."""

    id: str
    placed_at: str
    sport: str
    event: str
    market: str
    pick: str
    odds: float
    stake: float
    model_prob: float
    edge: float
    status: str = "pending"              # pending | won | lost | void | half_won | half_lost
    closing_odds: float | None = None
    pnl: float | None = None
    note: str = ""

    @property
    def clv(self) -> float | None:
        """Closing line value : combien on a battu la cloture, en %."""
        if not self.closing_odds or self.closing_odds <= 1.0:
            return None
        return (self.odds / self.closing_odds - 1.0) * 100.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        clv = self.clv
        d["clv_pct"] = None if clv is None else round(clv, 2)
        return d


def team_names(matches: Iterable[Match]) -> list[str]:
    seen: dict[str, None] = {}
    for m in matches:
        seen.setdefault(m.home, None)
        seen.setdefault(m.away, None)
    return sorted(seen)
