"""Le moteur de l'agent : entrainement, pronostic, detection de valeur.

Deux agents partagent la meme mecanique :
  FootballAgent  -> Dixon-Coles + Elo, tous les marches d'un match de foot.
  EsportsAgent   -> Elo par map + conversion vers la serie (Bo1/Bo3/Bo5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from .form import EsportsForm, TeamForm, esports_form, head_to_head, team_form
from .markets.esports import esports_markets
from .markets.football import football_markets
from .models import Fixture, MapResult, Match, Odds, Prediction, Selection
from .ratings.elo import EloConfig, EloRating
from .ratings.poisson import DixonColes, DixonColesConfig
from .reasoning import (
    RISK_NOTES,
    esports_context,
    explain_selection,
    football_context,
)
from .value import (
    StakingPlan,
    blend_distribution,
    confidence_label,
    devig,
    edge as compute_edge,
)


@dataclass
class AgentConfig:
    # Poids du modele face au marche (0 = suivre le marche, 1 = l'ignorer).
    weight_model: float = 0.45
    devig_method: str = "shin"
    staking: StakingPlan = field(default_factory=StakingPlan)
    # Marge minimale du bookmaker au-dela de laquelle on se mefie du prix.
    max_book_margin_pct: float = 12.0
    form_window: int = 10


class FootballAgent:
    """Pronostiqueur football base sur Dixon-Coles."""

    sport = "football"

    def __init__(
        self,
        config: AgentConfig | None = None,
        dc_config: DixonColesConfig | None = None,
        elo_config: EloConfig | None = None,
    ):
        self.config = config or AgentConfig()
        self.model = DixonColes(dc_config or DixonColesConfig())
        self.elo = EloRating(elo_config or EloConfig(k=20.0, home_advantage=60.0, mov_factor=0.6))
        self.matches: list[Match] = []

    # -- entrainement ------------------------------------------------------

    def fit(self, matches: Sequence[Match]) -> "FootballAgent":
        self.matches = sorted(matches, key=lambda m: m.date)
        self.model.fit(self.matches)
        self.elo = EloRating(self.elo.config).fit_matches(self.matches)
        return self

    def knows(self, *teams: str) -> list[str]:
        return [t for t in teams if not self.model.knows(t)]

    # -- pronostic ---------------------------------------------------------

    def predict(self, fixture: Fixture, neutral: bool = False) -> Prediction:
        unknown = self.knows(fixture.home, fixture.away)
        grid = self.model.score_matrix(fixture.home, fixture.away, neutral=neutral)
        probs = football_markets(grid)
        lam, mu = self.model.expected_goals(fixture.home, fixture.away, neutral=neutral)

        fh = team_form(self.matches, fixture.home, self.config.form_window)
        fa = team_form(self.matches, fixture.away, self.config.form_window)
        notes = football_context(
            fixture.home,
            fixture.away,
            lam,
            mu,
            fh,
            fa,
            self.model.strength(fixture.home),
            self.model.strength(fixture.away),
            self.elo.rating(fixture.home),
            self.elo.rating(fixture.away),
            head_to_head(self.matches, fixture.home, fixture.away),
            self.model.n_matches,
        )
        if unknown:
            notes.append(
                "ATTENTION : "
                + ", ".join(unknown)
                + " absent(s) de l'historique — probabilites peu fiables."
            )

        pred = Prediction(
            fixture=fixture,
            probabilities=probs,
            expected={
                "buts_domicile": round(lam, 3),
                "buts_exterieur": round(mu, 3),
                "total_buts": round(lam + mu, 3),
            },
            notes=notes + RISK_NOTES,
        )
        pred.selections = _evaluate(probs, fixture.odds, self.config, self.model.n_matches)
        return pred

    # -- persistance -------------------------------------------------------

    def to_dict(self) -> dict:
        return {"sport": self.sport, "model": self.model.to_dict(), "elo": self.elo.to_dict()}


class EsportsAgent:
    """Pronostiqueur esport : Elo par map, conversion vers la serie."""

    sport = "esport"

    def __init__(self, config: AgentConfig | None = None, elo_config: EloConfig | None = None):
        self.config = config or AgentConfig()
        # K plus eleve : les rosters bougent vite, la forme recente prime.
        self.elo = EloRating(elo_config or EloConfig(k=28.0, home_advantage=0.0, mov_factor=0.3))
        self.results: list[MapResult] = []
        # Amortissement : on tire les probas extremes vers 50% car un Elo
        # esport surestime les favoris (changements de roster, patchs, meta).
        self.shrink = 0.90

    def fit(self, results: Sequence[MapResult]) -> "EsportsAgent":
        self.results = sorted(results, key=lambda r: r.date)
        self.elo = EloRating(self.elo.config).fit_maps(self.results)
        return self

    def map_probability(self, a: str, b: str) -> float:
        p = self.elo.expected(a, b, neutral=True)
        return 0.5 + self.shrink * (p - 0.5)

    def predict(self, fixture: Fixture) -> Prediction:
        a, b = fixture.home, fixture.away
        p_map = self.map_probability(a, b)
        bo = max(1, fixture.best_of)
        raw = esports_markets(p_map, bo)

        # Les cles sont "A"/"B" cote modele : on les renomme avec les equipes.
        probs: dict[str, dict[str, float]] = {
            market: {_rename(key, a, b): val for key, val in values.items()}
            for market, values in raw.items()
        }

        fa = esports_form(self.results, a)
        fb = esports_form(self.results, b)
        p_series = raw["vainqueur"]["A"]
        notes = esports_context(
            a, b, p_map, p_series, bo,
            self.elo.rating(a), self.elo.rating(b), fa, fb, len(self.results),
        )
        unknown = [t for t in (a, b) if t not in self.elo.ratings]
        if unknown:
            notes.append(
                "ATTENTION : " + ", ".join(unknown)
                + " sans historique — Elo par defaut, pronostic non exploitable."
            )
        notes += [
            "L'esport bouge vite : verifiez les changements de roster, les "
            "remplacants et la patch note avant de miser.",
        ] + RISK_NOTES

        pred = Prediction(
            fixture=fixture,
            probabilities=probs,
            expected={
                "proba_map": round(p_map, 4),
                "proba_serie": round(p_series, 4),
                "best_of": bo,
            },
            notes=notes,
        )
        pred.selections = _evaluate(probs, fixture.odds, self.config, len(self.results))
        return pred

    def to_dict(self) -> dict:
        return {"sport": self.sport, "elo": self.elo.to_dict(), "shrink": self.shrink}


# --------------------------------------------------------------------------
# Confrontation modele / cotes
# --------------------------------------------------------------------------


def _rename(key: str, a: str, b: str) -> str:
    """Remplace les marqueurs generiques A/B par les noms d'equipes."""
    if key in ("A", "B"):
        return a if key == "A" else b
    if key.startswith("A ") or key.startswith("B "):
        return (a if key[0] == "A" else b) + key[1:]
    return key


def _evaluate(
    probs: dict[str, dict[str, float]],
    all_odds: Sequence[Odds],
    config: AgentConfig,
    sample_size: int,
) -> list[Selection]:
    """Pour chaque marche cote : devig, melange, edge, mise."""
    selections: list[Selection] = []
    for odds in all_odds:
        model_probs = probs.get(odds.market)
        if not model_probs:
            continue
        usable = {k: v for k, v in odds.prices.items() if k in model_probs and v > 1.0}
        if not usable:
            continue

        fair = devig(usable, config.devig_method) if len(usable) > 1 else {}
        margin = odds.margin_pct
        weight = config.weight_model
        if margin > config.max_book_margin_pct:
            # Prix charge : le signal du marche vaut moins, on s'appuie
            # davantage sur le modele mais on relevera le seuil d'edge.
            weight = min(1.0, weight + 0.15)

        subset = {k: model_probs[k] for k in usable}
        blended = blend_distribution(subset, fair or None, weight) if fair else subset

        for pick, price in usable.items():
            p = blended[pick]
            e = compute_edge(p, price)
            kelly, stake = config.staking.stake(p, price)
            sel = Selection(
                market=odds.market,
                pick=pick,
                model_prob=round(model_probs[pick], 4),
                odds=price,
                fair_prob=round(fair[pick], 4) if pick in fair else None,
                blended_prob=round(p, 4),
                edge=round(e, 4),
                kelly=round(kelly, 4),
                stake=stake,
                confidence=confidence_label(e, sample_size, bool(fair)),
            )
            sel.rationale = explain_selection(sel)
            if margin > config.max_book_margin_pct:
                sel.rationale.append(
                    f"Marge du bookmaker elevee ({margin:.1f}%) : prix peu fiable, "
                    "cherchez un meilleur operateur avant de miser."
                )
            selections.append(sel)

    selections.sort(key=lambda s: (s.stake or 0.0, s.edge or 0.0), reverse=True)
    return selections


def prediction_to_json(pred: Prediction, indent: int = 2) -> str:
    payload = {
        "rencontre": pred.fixture.label,
        "sport": pred.fixture.sport,
        "competition": pred.fixture.league,
        "attendus": pred.expected,
        "probabilites": {
            m: {k: round(v, 4) for k, v in sorted(vals.items(), key=lambda kv: -kv[1])}
            for m, vals in pred.probabilities.items()
        },
        "selections": [s.to_dict() for s in pred.selections],
        "analyse": pred.notes,
    }
    return json.dumps(payload, indent=indent, ensure_ascii=False)
