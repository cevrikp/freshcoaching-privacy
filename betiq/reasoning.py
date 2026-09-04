"""Generation du "pourquoi" : la justification lisible d'un pronostic."""

from __future__ import annotations

from .form import TeamForm, EsportsForm
from .models import Match, Selection


def football_context(
    home: str,
    away: str,
    lam: float,
    mu: float,
    fh: TeamForm,
    fa: TeamForm,
    strength_home: dict[str, float],
    strength_away: dict[str, float],
    elo_home: float,
    elo_away: float,
    h2h: list[Match],
    n_matches: int,
) -> list[str]:
    lines = [
        f"Buts attendus : {home} {lam:.2f} - {mu:.2f} {away} "
        f"(total {lam + mu:.2f}).",
        f"Forces modele : {home} attaque {strength_home['attaque']:.2f} / "
        f"defense {strength_home['defense']:.2f} — {away} attaque "
        f"{strength_away['attaque']:.2f} / defense {strength_away['defense']:.2f} "
        f"(1.00 = moyenne de la ligue).",
        f"Elo : {elo_home:.0f} contre {elo_away:.0f} "
        f"({elo_home - elo_away:+.0f} avant avantage du terrain).",
        f"Forme {home} : {fh.streak} sur {fh.played} matchs, {fh.ppg} pts/match, "
        f"{fh.gf_avg} but(s) marques et {fh.ga_avg} encaisses par match.",
        f"Forme {away} : {fa.streak} sur {fa.played} matchs, {fa.ppg} pts/match, "
        f"{fa.gf_avg} but(s) marques et {fa.ga_avg} encaisses par match.",
    ]
    if fh.home_played:
        lines.append(
            f"A domicile, {home} a gagne {fh.home_wins}/{fh.home_played} de ses "
            f"receptions recentes ; a l'exterieur {away} en a gagne "
            f"{fa.away_wins}/{fa.away_played}."
        )
    if h2h:
        detail = ", ".join(
            f"{m.date:%d/%m/%y} {m.home} {m.home_score}-{m.away_score} {m.away}"
            for m in h2h[-3:]
        )
        lines.append(f"Confrontations directes recentes : {detail}.")
    lines.append(
        f"Modele entraine sur {n_matches} matchs, ponderes par recence "
        "(les matchs vieux de six mois comptent moitie moins)."
    )
    return lines


def esports_context(
    a: str,
    b: str,
    p_map: float,
    p_series: float,
    best_of: int,
    elo_a: float,
    elo_b: float,
    fa: EsportsForm,
    fb: EsportsForm,
    n_maps: int,
) -> list[str]:
    fmt = "map unique" if best_of <= 1 else f"Bo{best_of}"
    return [
        f"Format : {fmt}. Probabilite par map {a} {p_map:.1%} — "
        f"probabilite de serie {p_series:.1%} "
        f"(le format amplifie l'avantage du favori).",
        f"Elo maps : {a} {elo_a:.0f} contre {b} {elo_b:.0f} ({elo_a - elo_b:+.0f}).",
        f"Forme {a} : {fa.streak} — {fa.won}/{fa.maps} maps gagnees ({fa.winrate_pct}%).",
        f"Forme {b} : {fb.streak} — {fb.won}/{fb.maps} maps gagnees ({fb.winrate_pct}%).",
        f"Modele entraine sur {n_maps} maps.",
    ]


def explain_selection(sel: Selection) -> list[str]:
    """Le raisonnement chiffre derriere une mise conseillee."""
    out: list[str] = []
    if sel.odds:
        out.append(
            f"Cote proposee {sel.odds:.2f} contre cote juste modele "
            f"{sel.fair_odds:.2f} : le bookmaker paie "
            f"{(sel.odds / sel.fair_odds - 1) * 100:+.1f}% par rapport a notre estimation."
        )
    if sel.fair_prob is not None:
        out.append(
            f"Marche (marge retiree) : {sel.fair_prob:.1%} — modele : "
            f"{sel.model_prob:.1%} — estimation retenue apres melange : "
            f"{sel.used_prob:.1%}."
        )
    if sel.edge is not None:
        out.append(f"Edge : {sel.edge:+.1%} d'esperance par euro mise.")
    if sel.stake:
        out.append(
            f"Mise conseillee : {sel.stake:.2f} EUR "
            f"({(sel.kelly or 0) * 100:.2f}% de bankroll, Kelly fractionne)."
        )
    else:
        out.append("Pas de mise : l'edge ne franchit pas le seuil ou la cote est hors plage.")
    return out


RISK_NOTES = [
    "Le modele ne connait ni les blessures, ni les suspensions, ni les rotations "
    "(coupe, match a enjeu nul) : verifiez la compo avant de valider.",
    "Un edge inferieur a 3-4% face a un bookmaker serieux est souvent du bruit "
    "de modele, pas de la valeur.",
    "Mesurez la CLV (cote prise contre cote de cloture) : c'est le seul indicateur "
    "fiable a court terme, le ROI demande des centaines de paris.",
]
