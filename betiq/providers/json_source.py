"""Rencontres et cotes au format JSON (le plus expressif : multi-marches)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from ..models import Fixture, Odds


def load_fixtures_json(path: str) -> list[Fixture]:
    """Lit un fichier de rencontres.

    Exemple :
    {
      "fixtures": [
        {"home": "Lyon", "away": "Nice", "league": "Ligue 1",
         "kickoff": "2026-09-12T20:45:00", "sport": "football",
         "odds": [
           {"market": "1X2", "bookmaker": "book", "prices": {"1":2.05,"X":3.5,"2":3.7}},
           {"market": "totals", "prices": {"+2.5":1.90,"-2.5":1.95}}
         ]}
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("fixtures", data) if isinstance(data, dict) else data
    return [_fixture(item) for item in raw]


def _fixture(item: dict[str, Any]) -> Fixture:
    kickoff = None
    if item.get("kickoff"):
        try:
            kickoff = datetime.fromisoformat(item["kickoff"])
        except ValueError:
            kickoff = None
    return Fixture(
        home=item["home"],
        away=item["away"],
        kickoff=kickoff,
        league=item.get("league", ""),
        sport=item.get("sport", "football"),
        best_of=int(item.get("best_of", 1)),
        odds=[
            Odds(
                market=o["market"],
                prices={k: float(v) for k, v in o["prices"].items()},
                bookmaker=o.get("bookmaker", ""),
            )
            for o in item.get("odds", [])
        ],
    )


def save_fixtures_json(path: str, fixtures: Iterable[Fixture]) -> None:
    payload = {
        "fixtures": [
            {
                "home": f.home,
                "away": f.away,
                "league": f.league,
                "sport": f.sport,
                "best_of": f.best_of,
                "kickoff": f.kickoff.isoformat() if f.kickoff else None,
                "odds": [
                    {"market": o.market, "bookmaker": o.bookmaker, "prices": o.prices}
                    for o in f.odds
                ],
            }
            for f in fixtures
        ]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
