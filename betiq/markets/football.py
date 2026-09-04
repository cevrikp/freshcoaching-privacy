"""Derivation de tous les marches football a partir de la grille de scores."""

from __future__ import annotations

from typing import Sequence

Grid = Sequence[Sequence[float]]

market_labels = {
    "1X2": "Resultat (1 / Nul / 2)",
    "double_chance": "Double chance",
    "btts": "Les deux equipes marquent",
    "totals": "Total de buts",
    "ah": "Handicap asiatique",
    "correct_score": "Score exact",
    "team_totals": "Total par equipe",
}


def _outcome_probs(grid: Grid) -> dict[str, float]:
    h = d = a = 0.0
    for i, row in enumerate(grid):
        for j, p in enumerate(row):
            if i > j:
                h += p
            elif i == j:
                d += p
            else:
                a += p
    return {"1": h, "X": d, "2": a}


def _btts(grid: Grid) -> dict[str, float]:
    yes = sum(p for i, row in enumerate(grid) for j, p in enumerate(row) if i and j)
    return {"oui": yes, "non": 1.0 - yes}


def _totals(grid: Grid, lines: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in lines:
        over = sum(
            p
            for i, row in enumerate(grid)
            for j, p in enumerate(row)
            if i + j > line
        )
        out[f"+{line:g}"] = over
        out[f"-{line:g}"] = 1.0 - over
    return out


def _team_totals(grid: Grid, lines: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in lines:
        home_over = sum(
            p for i, row in enumerate(grid) for p in [sum(row)] if i > line
        )
        away_over = sum(
            p
            for i, row in enumerate(grid)
            for j, p in enumerate(row)
            if j > line
        )
        out[f"dom +{line:g}"] = home_over
        out[f"dom -{line:g}"] = 1.0 - home_over
        out[f"ext +{line:g}"] = away_over
        out[f"ext -{line:g}"] = 1.0 - away_over
    return out


def _asian_handicap(grid: Grid, lines: Sequence[float]) -> dict[str, float]:
    """Probabilite equivalente d'un handicap asiatique (quarts de but inclus).

    Les remboursements (push) sont retires de l'enjeu : on renvoie
    W / (W + L), directement comparable a 1 / cote.
    """
    out: dict[str, float] = {}
    for line in lines:
        for side, sign in (("dom", 1.0), ("ext", -1.0)):
            handicap = sign * line or 0.0  # ligne miroir cote adverse (evite le "-0")
            win = push = 0.0
            for i, row in enumerate(grid):
                for j, p in enumerate(row):
                    margin = (i - j) if side == "dom" else (j - i)
                    for part, weight in _split_line(handicap):
                        adj = margin + part
                        if adj > 1e-9:
                            win += p * weight
                        elif abs(adj) <= 1e-9:
                            push += p * weight
            denom = max(1.0 - push, 1e-9)
            out[f"{side} {handicap:+g}"] = win / denom
    return out


def _split_line(line: float) -> list[tuple[float, float]]:
    """Un quart de but se joue moitie sur chaque demi-ligne encadrante."""
    if _quarter(line):
        return [(line - 0.25, 0.5), (line + 0.25, 0.5)]
    return [(line, 1.0)]


def _quarter(line: float) -> bool:
    return abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9


def _correct_score(grid: Grid, top: int = 8) -> dict[str, float]:
    scores = [
        (f"{i}-{j}", p)
        for i, row in enumerate(grid)
        for j, p in enumerate(row)
        if i <= 6 and j <= 6
    ]
    scores.sort(key=lambda kv: -kv[1])
    return dict(scores[:top])


def football_markets(
    grid: Grid,
    totals_lines: Sequence[float] = (0.5, 1.5, 2.5, 3.5, 4.5),
    ah_lines: Sequence[float] = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0, 1.5),
    team_total_lines: Sequence[float] = (0.5, 1.5, 2.5),
) -> dict[str, dict[str, float]]:
    """Tous les marches, prets a etre confrontes aux cotes du bookmaker."""
    o = _outcome_probs(grid)
    return {
        "1X2": o,
        "double_chance": {
            "1X": o["1"] + o["X"],
            "12": o["1"] + o["2"],
            "X2": o["X"] + o["2"],
        },
        "btts": _btts(grid),
        "totals": _totals(grid, totals_lines),
        "team_totals": _team_totals(grid, team_total_lines),
        "ah": _asian_handicap(grid, ah_lines),
        "correct_score": _correct_score(grid),
    }
