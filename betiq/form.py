"""Statistiques descriptives de forme, utilisees pour justifier un pronostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .models import Match, MapResult


@dataclass
class TeamForm:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_played: int = 0
    home_wins: int = 0
    away_played: int = 0
    away_wins: int = 0
    clean_sheets: int = 0
    btts: int = 0
    over25: int = 0
    last5: list[str] = field(default_factory=list)

    @property
    def ppg(self) -> float:
        return round((3 * self.wins + self.draws) / self.played, 2) if self.played else 0.0

    @property
    def gf_avg(self) -> float:
        return round(self.goals_for / self.played, 2) if self.played else 0.0

    @property
    def ga_avg(self) -> float:
        return round(self.goals_against / self.played, 2) if self.played else 0.0

    @property
    def btts_pct(self) -> float:
        return round(100 * self.btts / self.played, 0) if self.played else 0.0

    @property
    def over25_pct(self) -> float:
        return round(100 * self.over25 / self.played, 0) if self.played else 0.0

    @property
    def streak(self) -> str:
        return "".join(self.last5[-5:]) or "-"


def team_form(matches: Sequence[Match], team: str, last: int = 10) -> TeamForm:
    """Forme sur les `last` derniers matchs (du plus ancien au plus recent)."""
    rows = [m for m in matches if m.home == team or m.away == team]
    rows.sort(key=lambda m: m.date)
    rows = rows[-last:]
    f = TeamForm(team=team)
    for m in rows:
        at_home = m.home == team
        gf = m.home_score if at_home else m.away_score
        ga = m.away_score if at_home else m.home_score
        f.played += 1
        f.goals_for += gf
        f.goals_against += ga
        if at_home:
            f.home_played += 1
        else:
            f.away_played += 1
        if gf > ga:
            f.wins += 1
            f.last5.append("V")
            if at_home:
                f.home_wins += 1
            else:
                f.away_wins += 1
        elif gf == ga:
            f.draws += 1
            f.last5.append("N")
        else:
            f.losses += 1
            f.last5.append("D")
        if ga == 0:
            f.clean_sheets += 1
        if gf and ga:
            f.btts += 1
        if gf + ga > 2.5:
            f.over25 += 1
    return f


def head_to_head(matches: Sequence[Match], a: str, b: str, last: int = 6) -> list[Match]:
    rows = [
        m for m in matches
        if {m.home, m.away} == {a, b}
    ]
    rows.sort(key=lambda m: m.date)
    return rows[-last:]


@dataclass
class EsportsForm:
    team: str
    maps: int = 0
    won: int = 0
    last10: list[str] = field(default_factory=list)

    @property
    def winrate_pct(self) -> float:
        return round(100 * self.won / self.maps, 1) if self.maps else 0.0

    @property
    def streak(self) -> str:
        return "".join(self.last10[-6:]) or "-"


def esports_form(results: Sequence[MapResult], team: str, last: int = 20) -> EsportsForm:
    rows = [r for r in results if team in (r.winner, r.loser)]
    rows.sort(key=lambda r: r.date)
    rows = rows[-last:]
    f = EsportsForm(team=team)
    for r in rows:
        f.maps += 1
        if r.winner == team:
            f.won += 1
            f.last10.append("V")
        else:
            f.last10.append("D")
    return f
