"""Elo avec avantage du terrain et prise en compte de l'ecart de score.

Utilise pour :
  * l'esport (pas de match nul -> Elo est directement une proba de victoire) ;
  * le football, comme garde-fou / feature complementaire au modele Poisson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..models import Match, MapResult


@dataclass
class EloConfig:
    start: float = 1500.0
    k: float = 20.0
    # Avantage du terrain (football) ou de la side pick (esport), en points Elo.
    home_advantage: float = 60.0
    # Regression vers la moyenne appliquee entre deux saisons.
    season_regression: float = 0.25
    # Amplification du K selon la marge de victoire (0 = desactive).
    mov_factor: float = 0.0
    scale: float = 400.0


@dataclass
class EloRating:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)

    # -- acces -------------------------------------------------------------

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.start)

    def expected(self, home: str, away: str, neutral: bool = False) -> float:
        """Probabilite de victoire du 1er camp (pas de nul dans ce modele)."""
        ha = 0.0 if neutral else self.config.home_advantage
        diff = (self.rating(home) + ha) - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.config.scale))

    # -- mise a jour -------------------------------------------------------

    def update(
        self,
        home: str,
        away: str,
        score: float,
        margin: int = 0,
        neutral: bool = False,
    ) -> None:
        """`score` = 1 victoire domicile, 0.5 nul, 0 victoire exterieur."""
        exp = self.expected(home, away, neutral=neutral)
        k = self.config.k
        if self.config.mov_factor and margin > 1:
            # Multiplicateur log facon FiveThirtyEight : une large victoire
            # informe plus qu'une victoire a l'arrache, sans exploser le K.
            k *= 1.0 + self.config.mov_factor * math.log(margin)
        delta = k * (score - exp)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        self.games[home] = self.games.get(home, 0) + 1
        self.games[away] = self.games.get(away, 0) + 1

    # -- entrainement ------------------------------------------------------

    def fit_matches(self, matches: Sequence[Match], neutral: bool = False) -> "EloRating":
        last_year: int | None = None
        for m in sorted(matches, key=lambda x: x.date):
            if last_year is not None and m.date.year != last_year:
                self._regress()
            last_year = m.date.year
            score = {"H": 1.0, "D": 0.5, "A": 0.0}[m.result]
            self.update(
                m.home,
                m.away,
                score,
                margin=abs(m.home_score - m.away_score),
                neutral=neutral,
            )
        return self

    def fit_maps(self, maps: Iterable[MapResult]) -> "EloRating":
        """Esport : chaque map est une observation, terrain neutre."""
        for r in sorted(maps, key=lambda x: x.date):
            margin = 0
            if r.winner_score is not None and r.loser_score is not None:
                margin = abs(r.winner_score - r.loser_score)
            self.update(r.winner, r.loser, 1.0, margin=margin, neutral=True)
        return self

    def _regress(self) -> None:
        r = self.config.season_regression
        if r <= 0:
            return
        for team, val in self.ratings.items():
            self.ratings[team] = val + r * (self.config.start - val)

    # -- utilitaires -------------------------------------------------------

    def leaderboard(self, top: int | None = None) -> list[tuple[str, float, int]]:
        rows = [
            (t, round(v, 1), self.games.get(t, 0))
            for t, v in sorted(self.ratings.items(), key=lambda kv: -kv[1])
        ]
        return rows[:top] if top else rows

    def to_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "ratings": self.ratings,
            "games": self.games,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloRating":
        obj = cls(config=EloConfig(**data.get("config", {})))
        obj.ratings = dict(data.get("ratings", {}))
        obj.games = {k: int(v) for k, v in data.get("games", {}).items()}
        return obj
