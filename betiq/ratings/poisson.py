"""Modele Dixon-Coles (Poisson bivarie) pour le football.

Chaque equipe recoit une force offensive et une force defensive ; on ajoute
un avantage du terrain global. Les buts marques suivent :

    lambda (domicile) = exp(atk_dom - def_ext + avantage_terrain)
    mu     (exterieur) = exp(atk_ext - def_dom)

La correction Dixon-Coles ajuste les petits scores (0-0, 1-0, 0-1, 1-1) que
le Poisson independant modelise mal. Les matchs anciens sont ponderes par une
decroissance exponentielle : la forme recente compte davantage.

Implementation en Python pur (Adam sur la log-vraisemblance) : pas de numpy
requis, ce qui garde l'agent installable partout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from ..models import Match

MAX_GOALS = 12  # taille de la grille de scores exacts


@dataclass
class DixonColesConfig:
    # Demi-vie de la ponderation temporelle, en jours (180 = ~une demi-saison).
    half_life_days: float = 180.0
    # Regularisation L2 : ramene les forces vers 0 quand l'echantillon est mince.
    l2: float = 0.02
    iterations: int = 800
    learning_rate: float = 0.08
    # Grille de recherche pour la correction petits scores.
    rho_grid: tuple[float, ...] = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10)
    fit_rho: bool = True


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Correction Dixon-Coles sur les scores faibles."""
    if x == 0 and y == 0:
        return max(1.0 - lam * mu * rho, 1e-6)
    if x == 0 and y == 1:
        return max(1.0 + lam * rho, 1e-6)
    if x == 1 and y == 0:
        return max(1.0 + mu * rho, 1e-6)
    if x == 1 and y == 1:
        return max(1.0 - rho, 1e-6)
    return 1.0


@dataclass
class DixonColes:
    config: DixonColesConfig = field(default_factory=DixonColesConfig)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.25
    rho: float = -0.05
    base: float = 0.0            # log du nombre de buts moyen de la ligue
    n_matches: int = 0
    last_date: date | None = None

    # ------------------------------------------------------------------
    # Entrainement
    # ------------------------------------------------------------------

    def fit(self, matches: Sequence[Match], as_of: date | None = None) -> "DixonColes":
        matches = [m for m in matches if m.home and m.away]
        if not matches:
            raise ValueError("Aucun match fourni pour l'entrainement.")

        ref = as_of or max(m.date for m in matches)
        self.last_date = ref
        self.n_matches = len(matches)

        teams = sorted({m.home for m in matches} | {m.away for m in matches})
        self.attack = {t: 0.0 for t in teams}
        self.defence = {t: 0.0 for t in teams}

        decay = math.log(2.0) / max(self.config.half_life_days, 1.0)
        weights = [math.exp(-decay * max((ref - m.date).days, 0)) for m in matches]

        total_w = sum(weights) or 1.0
        goals = sum(w * (m.home_score + m.away_score) for w, m in zip(weights, matches))
        self.base = math.log(max(goals / (2.0 * total_w), 0.05))
        self.home_advantage = 0.25

        self._gradient_ascent(matches, weights)

        if self.config.fit_rho:
            self.rho = self._best_rho(matches, weights)
        return self

    def _gradient_ascent(self, matches: Sequence[Match], weights: Sequence[float]) -> None:
        cfg = self.config
        teams = list(self.attack)
        params = {f"a:{t}": 0.0 for t in teams}
        params.update({f"d:{t}": 0.0 for t in teams})
        params["home"] = self.home_advantage
        params["base"] = self.base

        m1 = {k: 0.0 for k in params}
        m2 = {k: 0.0 for k in params}
        b1, b2, eps = 0.9, 0.999, 1e-8

        for step in range(1, cfg.iterations + 1):
            grad = {k: 0.0 for k in params}
            for w, mt in zip(weights, matches):
                ah, dh = params[f"a:{mt.home}"], params[f"d:{mt.home}"]
                aa, da = params[f"a:{mt.away}"], params[f"d:{mt.away}"]
                lam = math.exp(params["base"] + ah - da + params["home"])
                mu = math.exp(params["base"] + aa - dh)
                lam = min(lam, 25.0)
                mu = min(mu, 25.0)
                rh = w * (mt.home_score - lam)
                ra = w * (mt.away_score - mu)
                grad[f"a:{mt.home}"] += rh
                grad[f"d:{mt.away}"] -= rh
                grad[f"a:{mt.away}"] += ra
                grad[f"d:{mt.home}"] -= ra
                grad["home"] += rh
                grad["base"] += rh + ra

            for t in teams:  # regularisation L2 vers 0
                grad[f"a:{t}"] -= cfg.l2 * params[f"a:{t}"]
                grad[f"d:{t}"] -= cfg.l2 * params[f"d:{t}"]

            for k, g in grad.items():
                m1[k] = b1 * m1[k] + (1 - b1) * g
                m2[k] = b2 * m2[k] + (1 - b2) * g * g
                mhat = m1[k] / (1 - b1 ** step)
                vhat = m2[k] / (1 - b2 ** step)
                params[k] += cfg.learning_rate * mhat / (math.sqrt(vhat) + eps)

            # Identifiabilite : moyenne des forces recentree a chaque pas.
            for prefix in ("a", "d"):
                keys = [f"{prefix}:{t}" for t in teams]
                mean = sum(params[k] for k in keys) / len(keys)
                for k in keys:
                    params[k] -= mean

        self.attack = {t: params[f"a:{t}"] for t in teams}
        self.defence = {t: params[f"d:{t}"] for t in teams}
        self.home_advantage = params["home"]
        self.base = params["base"]

    def _best_rho(self, matches: Sequence[Match], weights: Sequence[float]) -> float:
        best, best_ll = 0.0, -float("inf")
        for rho in self.config.rho_grid:
            ll = 0.0
            for w, mt in zip(weights, matches):
                lam, mu = self.expected_goals(mt.home, mt.away)
                p = (
                    dc_tau(mt.home_score, mt.away_score, lam, mu, rho)
                    * _poisson_pmf(mt.home_score, lam)
                    * _poisson_pmf(mt.away_score, mu)
                )
                ll += w * math.log(max(p, 1e-12))
            if ll > best_ll:
                best, best_ll = rho, ll
        return best

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def knows(self, team: str) -> bool:
        return team in self.attack

    def expected_goals(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float]:
        """Buts attendus (lambda domicile, mu exterieur)."""
        ah = self.attack.get(home, 0.0)
        dh = self.defence.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        da = self.defence.get(away, 0.0)
        ha = 0.0 if neutral else self.home_advantage
        lam = math.exp(self.base + ah - da + ha)
        mu = math.exp(self.base + aa - dh)
        return min(lam, 8.0), min(mu, 8.0)

    def score_matrix(
        self, home: str, away: str, neutral: bool = False, max_goals: int = MAX_GOALS
    ) -> list[list[float]]:
        """Grille normalisee des probabilites de score exact."""
        lam, mu = self.expected_goals(home, away, neutral=neutral)
        ph = [_poisson_pmf(i, lam) for i in range(max_goals + 1)]
        pa = [_poisson_pmf(j, mu) for j in range(max_goals + 1)]
        grid = [
            [ph[i] * pa[j] * dc_tau(i, j, lam, mu, self.rho) for j in range(max_goals + 1)]
            for i in range(max_goals + 1)
        ]
        total = sum(sum(row) for row in grid) or 1.0
        return [[v / total for v in row] for row in grid]

    def strength(self, team: str) -> dict[str, float]:
        """Lecture humaine : > 1 = au-dessus de la moyenne de la ligue."""
        return {
            "attaque": round(math.exp(self.attack.get(team, 0.0)), 3),
            "defense": round(math.exp(self.defence.get(team, 0.0)), 3),
        }

    def to_dict(self) -> dict:
        return {
            "config": {**self.config.__dict__, "rho_grid": list(self.config.rho_grid)},
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "base": self.base,
            "n_matches": self.n_matches,
            "last_date": self.last_date.isoformat() if self.last_date else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DixonColes":
        cfg = dict(data.get("config", {}))
        cfg["rho_grid"] = tuple(cfg.get("rho_grid", DixonColesConfig().rho_grid))
        obj = cls(config=DixonColesConfig(**cfg))
        obj.attack = dict(data["attack"])
        obj.defence = dict(data["defence"])
        obj.home_advantage = float(data["home_advantage"])
        obj.rho = float(data["rho"])
        obj.base = float(data["base"])
        obj.n_matches = int(data.get("n_matches", 0))
        ld = data.get("last_date")
        obj.last_date = date.fromisoformat(ld) if ld else None
        return obj
