"""Lecture de donnees depuis des CSV.

Trois formats sont acceptes :
  * format simple      : date,home,away,home_score,away_score[,league]
  * format football-data.co.uk (gratuit, historique + cotes de cloture)
  * format esport      : date,winner,loser[,game,event,winner_score,loser_score]
                         ou date,team_a,team_b,maps_a,maps_b (serie -> maps)
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from typing import Any, Iterable

from ..models import Fixture, MapResult, Match, Odds

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%d-%m-%Y")


def parse_date(value: str) -> date:
    value = (value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Date illisible : {value!r}")


def _rows(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def _get(row: dict[str, Any], *names: str) -> str:
    for n in names:
        if row.get(n):
            return row[n]
    return ""


def _float(value: str) -> float | None:
    try:
        f = float(value)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Football
# --------------------------------------------------------------------------


def load_matches_csv(path: str, league: str = "") -> list[Match]:
    """Format simple ; tolere aussi les entetes football-data.co.uk."""
    out: list[Match] = []
    for row in _rows(path):
        home = _get(row, "home", "HomeTeam", "home_team")
        away = _get(row, "away", "AwayTeam", "away_team")
        hs = _get(row, "home_score", "FTHG", "hg")
        as_ = _get(row, "away_score", "FTAG", "ag")
        if not (home and away and hs != "" and as_ != ""):
            continue
        closing = _closing_odds(row)
        out.append(
            Match(
                date=parse_date(_get(row, "date", "Date")),
                home=home,
                away=away,
                home_score=int(float(hs)),
                away_score=int(float(as_)),
                league=_get(row, "league", "Div") or league,
                closing_odds=closing,
            )
        )
    out.sort(key=lambda m: m.date)
    return out


def _closing_odds(row: dict[str, Any]) -> tuple[float, float, float] | None:
    """Cotes de cloture Pinnacle si presentes, sinon Bet365 / moyenne marche."""
    for h, d, a in (
        ("PSCH", "PSCD", "PSCA"),   # Pinnacle closing
        ("B365CH", "B365CD", "B365CA"),
        ("AvgCH", "AvgCD", "AvgCA"),
        ("PSH", "PSD", "PSA"),
        ("B365H", "B365D", "B365A"),
        ("odds_1", "odds_x", "odds_2"),
    ):
        vals = (_float(row.get(h, "")), _float(row.get(d, "")), _float(row.get(a, "")))
        if all(vals):
            return vals  # type: ignore[return-value]
    return None


def load_football_data_uk(path: str) -> list[Match]:
    """Alias explicite pour les CSV de football-data.co.uk."""
    return load_matches_csv(path)


def load_fixtures_csv(path: str, sport: str = "football") -> list[Fixture]:
    """Rencontres a venir + cotes 1X2 (ou vainqueur esport) optionnelles.

    Colonnes reconnues : date,home,away,league,best_of,
    odds_1,odds_x,odds_2 (football) ou odds_home,odds_away (esport),
    ou_line,odds_over,odds_under,odds_btts_yes,odds_btts_no.
    """
    fixtures: list[Fixture] = []
    for row in _rows(path):
        home = _get(row, "home", "team_a", "HomeTeam")
        away = _get(row, "away", "team_b", "AwayTeam")
        if not (home and away):
            continue
        kickoff = None
        raw_date = _get(row, "date", "Date", "kickoff")
        if raw_date:
            try:
                kickoff = datetime.combine(parse_date(raw_date), datetime.min.time())
            except ValueError:
                kickoff = None
        fx = Fixture(
            home=home,
            away=away,
            kickoff=kickoff,
            league=_get(row, "league", "Div", "event"),
            sport=_get(row, "sport") or sport,
            best_of=int(_get(row, "best_of") or 1),
        )
        fx.odds = _fixture_odds(row, fx)
        fixtures.append(fx)
    return fixtures


def _fixture_odds(row: dict[str, Any], fx: Fixture) -> list[Odds]:
    book = _get(row, "bookmaker") or "csv"
    odds: list[Odds] = []

    o1, ox, o2 = (_float(row.get(k, "")) for k in ("odds_1", "odds_x", "odds_2"))
    if o1 and ox and o2:
        odds.append(Odds("1X2", {"1": o1, "X": ox, "2": o2}, book))

    oh, oa = _float(row.get("odds_home", "")), _float(row.get("odds_away", ""))
    if oh and oa:
        odds.append(Odds("vainqueur", {fx.home: oh, fx.away: oa}, book))

    line = row.get("ou_line") or "2.5"
    over, under = _float(row.get("odds_over", "")), _float(row.get("odds_under", ""))
    if over and under:
        key = "total_maps" if fx.sport == "esport" else "totals"
        odds.append(Odds(key, {f"+{float(line):g}": over, f"-{float(line):g}": under}, book))

    yes, no = _float(row.get("odds_btts_yes", "")), _float(row.get("odds_btts_no", ""))
    if yes and no:
        odds.append(Odds("btts", {"oui": yes, "non": no}, book))

    return odds


# --------------------------------------------------------------------------
# Esport
# --------------------------------------------------------------------------


def load_maps_csv(path: str, game: str = "") -> list[MapResult]:
    """Resultats esport, map par map ou serie a eclater en maps."""
    out: list[MapResult] = []
    for row in _rows(path):
        d = parse_date(_get(row, "date", "Date"))
        game_name = _get(row, "game", "jeu") or game
        event = _get(row, "event", "tournoi")
        winner, loser = _get(row, "winner"), _get(row, "loser")
        if winner and loser:
            out.append(
                MapResult(
                    date=d,
                    winner=winner,
                    loser=loser,
                    game=game_name,
                    event=event,
                    winner_score=_int_or_none(row.get("winner_score")),
                    loser_score=_int_or_none(row.get("loser_score")),
                )
            )
            continue
        # Format serie : on eclate 2-1 en trois observations de map.
        a, b = _get(row, "team_a", "home"), _get(row, "team_b", "away")
        ma, mb = _int_or_none(_get(row, "maps_a", "score_a")), _int_or_none(
            _get(row, "maps_b", "score_b")
        )
        if not (a and b and ma is not None and mb is not None):
            continue
        for _ in range(ma):
            out.append(MapResult(date=d, winner=a, loser=b, game=game_name, event=event))
        for _ in range(mb):
            out.append(MapResult(date=d, winner=b, loser=a, game=game_name, event=event))
    out.sort(key=lambda r: r.date)
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def write_matches_csv(path: str, matches: Iterable[Match]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home", "away", "home_score", "away_score", "league"])
        for m in matches:
            w.writerow([m.date.isoformat(), m.home, m.away, m.home_score, m.away_score, m.league])
