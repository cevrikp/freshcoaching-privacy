"""Comparaison modele / marche : retrait de la marge, edge et mise Kelly.

Le coeur du metier d'un parieur pro tient en trois gestes :
  1. retirer la marge du bookmaker pour obtenir la vraie proba du marche ;
  2. melanger sa propre estimation avec celle du marche (le marche est fort,
     l'ignorer est la premiere source de ruine) ;
  3. ne miser que si l'edge survit a ce melange, et miser petit (Kelly fractionne).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


# --------------------------------------------------------------------------
# Retrait de la marge (devig)
# --------------------------------------------------------------------------


def implied(prices: Mapping[str, float]) -> dict[str, float]:
    return {k: 1.0 / v for k, v in prices.items() if v and v > 1.0}


def devig(prices: Mapping[str, float], method: str = "shin") -> dict[str, float]:
    """Probabilites 'justes' du marche, marge retiree.

    - `multiplicative` : division par la somme. Simple, mais surestime les
      favoris car elle retire la marge proportionnellement.
    - `power`          : q_i^k avec somme = 1. Retire plus de marge aux outsiders.
    - `shin`           : modele de Shin (parieurs informes). Reference du milieu.
    """
    q = implied(prices)
    if not q:
        return {}
    if len(q) == 1:
        return {k: min(v, 1.0) for k, v in q.items()}
    method = method.lower()
    if method in ("mult", "multiplicative", "proportional"):
        return _normalise(q)
    if method == "power":
        return _power(q)
    if method == "shin":
        return _shin(q)
    raise ValueError(f"Methode de devig inconnue : {method}")


def _normalise(q: Mapping[str, float]) -> dict[str, float]:
    s = sum(q.values()) or 1.0
    return {k: v / s for k, v in q.items()}


def _power(q: Mapping[str, float]) -> dict[str, float]:
    lo, hi = 0.5, 2.0
    for _ in range(80):
        k = (lo + hi) / 2
        s = sum(v**k for v in q.values())
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return _normalise({key: v**k for key, v in q.items()})


def _shin(q: Mapping[str, float]) -> dict[str, float]:
    """Resolution du z de Shin par bissection."""
    total = sum(q.values())
    if total <= 1.0:
        return _normalise(q)
    lo, hi = 0.0, 0.5
    for _ in range(100):
        z = (lo + hi) / 2
        s = sum(_shin_prob(v, z, total) for v in q.values())
        if s > 1.0:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2
    return _normalise({k: _shin_prob(v, z, total) for k, v in q.items()})


def _shin_prob(qi: float, z: float, total: float) -> float:
    inner = z * z + 4.0 * (1.0 - z) * qi * qi / total
    return (math.sqrt(max(inner, 0.0)) - z) / (2.0 * (1.0 - z))


# --------------------------------------------------------------------------
# Melange modele / marche
# --------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend(model_p: float, market_p: float | None, weight_model: float = 0.5) -> float:
    """Melange en espace logit (plus stable aux extremes qu'une moyenne simple)."""
    if market_p is None:
        return model_p
    w = min(max(weight_model, 0.0), 1.0)
    return _sigmoid(w * _logit(model_p) + (1 - w) * _logit(market_p))


def blend_distribution(
    model: Mapping[str, float],
    market: Mapping[str, float] | None,
    weight_model: float = 0.5,
) -> dict[str, float]:
    """Melange puis renormalise sur l'ensemble des issues d'un marche."""
    if not market:
        return dict(model)
    out = {k: blend(v, market.get(k), weight_model) for k, v in model.items()}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


# --------------------------------------------------------------------------
# Edge et mise
# --------------------------------------------------------------------------


def edge(prob: float, odds: float) -> float:
    """Esperance par euro mise : > 0 = pari a valeur positive."""
    return prob * odds - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    """Kelly plein : fraction de bankroll qui maximise la croissance log."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * odds - 1.0) / b
    return max(f, 0.0)


@dataclass
class StakingPlan:
    bankroll: float = 1000.0
    kelly_fraction_used: float = 0.25   # quart de Kelly : standard chez les pros
    max_stake_pct: float = 0.02         # jamais plus de 2% de bankroll sur un pari
    min_edge: float = 0.03              # 3% d'edge minimum apres melange
    min_odds: float = 1.30
    max_odds: float = 10.0
    round_to: float = 0.5

    def stake(self, prob: float, odds: float) -> tuple[float, float]:
        """Renvoie (fraction de bankroll, mise en euros)."""
        if odds < self.min_odds or odds > self.max_odds:
            return 0.0, 0.0
        if edge(prob, odds) < self.min_edge:
            return 0.0, 0.0
        f = kelly_fraction(prob, odds) * self.kelly_fraction_used
        f = min(f, self.max_stake_pct)
        amount = f * self.bankroll
        if self.round_to:
            amount = round(amount / self.round_to) * self.round_to
        return f, round(amount, 2)


def confidence_label(edge_value: float, sample_size: int, blended: bool) -> str:
    """Etiquette lisible : l'edge seul ne suffit pas, la taille d'echantillon compte."""
    if sample_size < 60:
        return "faible (echantillon court)"
    if edge_value >= 0.10 and blended:
        return "elevee"
    if edge_value >= 0.05:
        return "correcte"
    if edge_value > 0:
        return "moyenne"
    return "nulle"


def best_price(quotes: Iterable[float]) -> float | None:
    prices = [q for q in quotes if q and q > 1.0]
    return max(prices) if prices else None
