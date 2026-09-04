"""Recuperation des cotes en direct via the-odds-api.com (cle gratuite).

Optionnel : tout le reste de l'agent fonctionne hors ligne avec des CSV/JSON.
On utilise urllib (stdlib) pour ne rien imposer comme dependance.

    export ODDS_API_KEY="votre_cle"
    betiq odds --sport soccer_france_ligue_one --out fixtures.json
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from ..models import Fixture, Odds

BASE = "https://api.the-odds-api.com/v4"

# Correspondance cle the-odds-api -> marche interne BetIQ.
_MARKET_MAP = {"h2h": "1X2", "totals": "totals"}


class TheOddsAPI:
    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "Cle API absente : definissez ODDS_API_KEY ou passez --api-key."
            )

    def _get(self, path: str, **params: Any) -> Any:
        params["apiKey"] = self.api_key
        url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # message lisible plutot qu'une trace
            raise RuntimeError(
                f"the-odds-api a repondu {exc.code} : {exc.read().decode('utf-8')[:200]}"
            ) from exc

    def sports(self) -> list[dict[str, Any]]:
        return self._get("/sports")

    def odds(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
    ) -> list[Fixture]:
        raw = self._get(
            f"/sports/{sport_key}/odds",
            regions=regions,
            markets=markets,
            oddsFormat=odds_format,
        )
        return [self._to_fixture(ev, sport_key) for ev in raw]

    def _to_fixture(self, event: dict[str, Any], sport_key: str) -> Fixture:
        home, away = event.get("home_team", ""), event.get("away_team", "")
        is_esport = "esport" in sport_key or sport_key.startswith(("csgo", "lol", "dota"))
        fx = Fixture(
            home=home,
            away=away,
            kickoff=_parse_iso(event.get("commence_time")),
            league=event.get("sport_title", sport_key),
            sport="esport" if is_esport else "football",
        )
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                key = _MARKET_MAP.get(market.get("key", ""))
                if not key:
                    continue
                prices = _prices(market, home, away, key, is_esport)
                if prices:
                    fx.odds.append(
                        Odds(
                            market="vainqueur" if (is_esport and key == "1X2") else key,
                            prices=prices,
                            bookmaker=book.get("title", ""),
                        )
                    )
        return fx


def _prices(
    market: dict[str, Any], home: str, away: str, key: str, is_esport: bool
) -> dict[str, float]:
    out: dict[str, float] = {}
    for oc in market.get("outcomes", []):
        name, price = oc.get("name", ""), float(oc.get("price", 0) or 0)
        if price <= 1.0:
            continue
        if key == "1X2":
            if is_esport:
                out[name] = price
            elif name == home:
                out["1"] = price
            elif name == away:
                out["2"] = price
            elif name.lower() in ("draw", "tie", "nul"):
                out["X"] = price
        elif key == "totals":
            point = oc.get("point")
            if point is None:
                continue
            sign = "+" if name.lower() == "over" else "-"
            out[f"{sign}{float(point):g}"] = price
    return out


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
