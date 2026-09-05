"""Genere le jeu de donnees de demonstration (100% synthetique).

Aucune donnee reelle n'est utilisee : les equipes sont fictives, les cotes
sont simulees a partir des vraies probabilites du generateur, majorees d'une
marge de bookmaker et d'un bruit d'estimation. Cela permet de faire tourner
l'agent hors ligne, de tester le backtest et de verifier que la detection de
valeur fonctionne.

    python3 scripts/generate_demo_data.py
"""

from __future__ import annotations

import csv
import math
import os
import random
from datetime import date, timedelta

random.seed(20260904)
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo")

FOOT_TEAMS = {
    "Rouen FC": (1.55, 0.80), "Annecy SC": (1.40, 0.85), "Vannes AC": (1.30, 0.90),
    "Colmar United": (1.20, 0.95), "Biarritz OL": (1.15, 1.00), "Perpignan FC": (1.05, 1.00),
    "Chartres SC": (1.00, 1.05), "Tarbes AS": (0.95, 1.10), "Laval CF": (0.90, 1.10),
    "Quimper FC": (0.85, 1.15), "Albi SC": (0.80, 1.20), "Verdun AC": (0.72, 1.30),
}
HOME_BOOST, BASE_GOALS = 1.20, 1.32

ESPORT_TEAMS = {
    "Nova Esports": 1720, "Vertex Gaming": 1660, "Solstice": 1600, "Ironclad": 1560,
    "Meridian": 1520, "Kappa Squad": 1470, "Nightfall": 1420, "Zenith Nine": 1360,
}


def poisson(lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p <= limit:
            return k
        k += 1


def true_1x2(lh: float, la: float, max_goals: int = 10) -> tuple[float, float, float]:
    def pmf(k, l):
        return math.exp(-l + k * math.log(l) - math.lgamma(k + 1))
    h = d = a = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf(i, lh) * pmf(j, la)
            h, d, a = (h + p, d, a) if i > j else (h, d + p, a) if i == j else (h, d, a + p)
    s = h + d + a
    return h / s, d / s, a / s


def book_odds(probs: tuple[float, float, float], margin: float = 0.055) -> list[float]:
    """Cotes de cloture : proba vraie + biais d'estimation + marge."""
    noisy = [max(p * math.exp(random.gauss(0, 0.09)), 0.01) for p in probs]
    total = sum(noisy)
    return [round(max(1.0 / (p / total * (1 + margin)), 1.02), 2) for p in noisy]


def football_rows(seasons: int = 5) -> list[list]:
    rows, teams = [], list(FOOT_TEAMS)
    day = date(2021, 8, 5)
    for _ in range(seasons):
        pairs = [(h, a) for h in teams for a in teams if h != a]
        random.shuffle(pairs)
        for idx, (h, a) in enumerate(pairs):
            ah, dh = FOOT_TEAMS[h]
            aa, da = FOOT_TEAMS[a]
            lh = BASE_GOALS * ah * da * HOME_BOOST
            la = BASE_GOALS * aa * dh
            hs, as_ = poisson(lh), poisson(la)
            o1, ox, o2 = book_odds(true_1x2(lh, la))
            rows.append(
                [(day + timedelta(days=idx // 6 * 7 + idx % 6)).isoformat(),
                 h, a, hs, as_, "Ligue Demo", o1, ox, o2]
            )
        day += timedelta(days=300)
    rows.sort(key=lambda r: r[0])
    return rows


def esport_rows(n: int = 1600) -> list[list]:
    rows, teams = [], list(ESPORT_TEAMS)
    day = date(2025, 1, 6)
    for k in range(n):
        a, b = random.sample(teams, 2)
        pa = 1 / (1 + 10 ** (-(ESPORT_TEAMS[a] - ESPORT_TEAMS[b]) / 400))
        w, l = (a, b) if random.random() < pa else (b, a)
        ws, ls = (13, random.randint(2, 11))
        rows.append([(day + timedelta(days=k // 5)).isoformat(), w, l, "CS2", "Demo League", ws, ls])
    return rows


def write(path: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path} : {len(rows)} lignes")


def main() -> None:
    foot = football_rows()
    write(
        os.path.join(ROOT, "football_resultats.csv"),
        ["date", "home", "away", "home_score", "away_score", "league", "odds_1", "odds_x", "odds_2"],
        foot,
    )
    write(
        os.path.join(ROOT, "esport_maps.csv"),
        ["date", "winner", "loser", "game", "event", "winner_score", "loser_score"],
        esport_rows(),
    )

    # Rencontres a venir : cotes volontairement decalees sur quelques matchs
    # pour que la detection de valeur ait quelque chose a trouver.
    upcoming = [
        ("Chartres SC", "Annecy SC"), ("Laval CF", "Vannes AC"), ("Biarritz OL", "Colmar United"),
        ("Perpignan FC", "Rouen FC"), ("Quimper FC", "Tarbes AS"), ("Albi SC", "Verdun AC"),
    ]
    rows = []
    kickoff = date(2026, 9, 12)
    for i, (h, a) in enumerate(upcoming):
        ah, dh = FOOT_TEAMS[h]
        aa, da = FOOT_TEAMS[a]
        lh, la = BASE_GOALS * ah * da * HOME_BOOST, BASE_GOALS * aa * dh
        o1, ox, o2 = book_odds(true_1x2(lh, la), margin=0.05)
        if i % 2 == 0:  # cote allongee sur l'exterieur -> value a trouver
            o2 = round(o2 * 1.16, 2)
        rows.append([(kickoff + timedelta(days=i % 3)).isoformat(), h, a, "Ligue Demo",
                     o1, ox, o2, 2.5, 1.92, 1.90])
    write(
        os.path.join(ROOT, "football_a_venir.csv"),
        ["date", "home", "away", "league", "odds_1", "odds_x", "odds_2",
         "ou_line", "odds_over", "odds_under"],
        rows,
    )

    es_rows = []
    for i, (a, b) in enumerate([("Vertex Gaming", "Solstice"), ("Meridian", "Ironclad"),
                                ("Kappa Squad", "Nightfall"), ("Nova Esports", "Vertex Gaming")]):
        pa = 1 / (1 + 10 ** (-(ESPORT_TEAMS[a] - ESPORT_TEAMS[b]) / 400))
        ps = pa**2 * (1 + 2 * (1 - pa))  # Bo3
        oa = round(max(1 / (ps * 1.045), 1.02), 2)
        ob = round(max(1 / ((1 - ps) * 1.045), 1.02), 2)
        if i == 1:
            ob = round(ob * 1.15, 2)
        es_rows.append([(date(2026, 9, 13) + timedelta(days=i)).isoformat(), a, b,
                        "Demo League", 3, "esport", oa, ob])
    write(
        os.path.join(ROOT, "esport_a_venir.csv"),
        ["date", "home", "away", "league", "best_of", "sport", "odds_home", "odds_away"],
        es_rows,
    )


if __name__ == "__main__":
    main()
