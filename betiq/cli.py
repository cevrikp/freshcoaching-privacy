"""Interface en ligne de commande de BetIQ."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Sequence

from .backtest import backtest_football
from .bankroll import Bankroll, summarise
from .engine import AgentConfig, EsportsAgent, FootballAgent, prediction_to_json
from .models import Fixture, Odds, Prediction
from .providers import (
    load_fixtures_csv,
    load_fixtures_json,
    load_maps_csv,
    load_matches_csv,
    save_fixtures_json,
)
from .value import StakingPlan

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_FOOT = os.path.join(HERE, "data", "demo", "football_resultats.csv")
DEMO_ESPORT = os.path.join(HERE, "data", "demo", "esport_maps.csv")
DEMO_FIXTURES = os.path.join(HERE, "data", "demo", "football_a_venir.csv")
DEMO_ESPORT_FIXTURES = os.path.join(HERE, "data", "demo", "esport_a_venir.csv")

BANNER = "BetIQ — pronostics statistiques et detection de valeur"
DISCLAIMER = (
    "Rappel : parier comporte un risque de perte. Aucun modele ne garantit un gain ; "
    "l'objectif est un avantage faible sur un grand nombre de paris. Ne misez que ce "
    "que vous pouvez perdre. Aide : joueurs-info-service.fr — 09 74 75 13 13."
)


# --------------------------------------------------------------------------
# Aides
# --------------------------------------------------------------------------


def _staking(args: argparse.Namespace) -> StakingPlan:
    return StakingPlan(
        bankroll=args.bankroll,
        kelly_fraction_used=args.kelly,
        max_stake_pct=args.max_stake,
        min_edge=args.min_edge,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )


def _config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        weight_model=args.weight_model,
        devig_method=args.devig,
        staking=_staking(args),
    )


def _parse_odds(spec: str | None, market: str) -> Odds | None:
    """--odds "1=2.10,X=3.40,2=3.60" ou "T1=1.35,KOI=3.20"."""
    if not spec:
        return None
    prices: dict[str, float] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise SystemExit(f"Format de cote invalide : {part!r} (attendu selection=cote)")
        key, value = part.split("=", 1)
        prices[key.strip()] = float(value)
    return Odds(market=market, prices=prices, bookmaker="saisie manuelle")


def _fit_football(args: argparse.Namespace) -> FootballAgent:
    matches = load_matches_csv(args.results)
    if not matches:
        raise SystemExit(f"Aucun match lisible dans {args.results}")
    return FootballAgent(config=_config(args)).fit(matches)


def _fit_esport(args: argparse.Namespace) -> EsportsAgent:
    maps = load_maps_csv(args.results)
    if not maps:
        raise SystemExit(f"Aucune map lisible dans {args.results}")
    return EsportsAgent(config=_config(args)).fit(maps)


def _load_fixtures(path: str, sport: str) -> list[Fixture]:
    if path.endswith(".json"):
        return load_fixtures_json(path)
    return load_fixtures_csv(path, sport=sport)


# --------------------------------------------------------------------------
# Rendu texte
# --------------------------------------------------------------------------


def render_prediction(pred: Prediction, top_markets: Sequence[str] | None = None) -> str:
    fx = pred.fixture
    out = [
        "",
        f"=== {fx.label} ===",
        f"{fx.league or fx.sport}"
        + (f" — {fx.kickoff:%d/%m/%Y %H:%M}" if fx.kickoff else "")
        + (f" — Bo{fx.best_of}" if fx.sport == "esport" and fx.best_of > 1 else ""),
        "",
        "Probabilites du modele",
    ]
    markets = top_markets or list(pred.probabilities)
    for market in markets:
        values = pred.probabilities.get(market)
        if not values:
            continue
        if market in ("totals", "total_maps"):
            # Lignes de total : on affiche les "plus de X", triees par ligne.
            items = sorted(
                ((k, v) for k, v in values.items() if k.startswith("+")),
                key=lambda kv: float(kv[0][1:]),
            )[:6]
        else:
            items = sorted(values.items(), key=lambda kv: -kv[1])[:6]
        line = "   ".join(f"{k} {v:.1%}" for k, v in items)
        out.append(f"  {market:<15} {line}")

    if pred.expected:
        out.append("")
        out.append(
            "Attendus : "
            + ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in pred.expected.items())
        )

    out += ["", "Analyse"]
    out += [f"  - {n}" for n in pred.notes]

    playable = [s for s in pred.selections if s.stake]
    out += ["", "Paris a valeur detectes"]
    if not playable:
        out.append("  Aucun : le marche est correctement price, on passe son tour.")
    for s in playable:
        out.append(
            f"  > {s.market} / {s.pick} @ {s.odds:.2f} — edge {s.edge:+.1%} — "
            f"mise {s.stake:.2f} EUR — confiance {s.confidence}"
        )
        out += [f"      {r}" for r in s.rationale]

    others = [s for s in pred.selections if not s.stake and (s.edge or -1) > 0][:3]
    if others:
        out.append("")
        out.append("Edges positifs mais sous le seuil (pour information)")
        for s in others:
            out.append(f"  . {s.market} / {s.pick} @ {s.odds:.2f} — edge {s.edge:+.1%}")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Commandes
# --------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    if args.sport == "football":
        agent = _fit_football(args)
        odds = _parse_odds(args.odds, "1X2")
        fx = Fixture(home=args.home, away=args.away, sport="football", league=args.league)
        if odds:
            fx.odds.append(odds)
        if args.odds_ou:
            fx.odds.append(_parse_odds(args.odds_ou, "totals"))
        pred = agent.predict(fx, neutral=args.neutral)
        markets = ["1X2", "double_chance", "totals", "btts", "correct_score"]
    else:
        agent = _fit_esport(args)
        odds = _parse_odds(args.odds, "vainqueur")
        fx = Fixture(
            home=args.home, away=args.away, sport="esport",
            league=args.league, best_of=args.best_of,
        )
        if odds:
            fx.odds.append(odds)
        pred = agent.predict(fx)
        markets = ["vainqueur", "score_serie", "handicap_maps", "total_maps"]

    print(prediction_to_json(pred) if args.json else render_prediction(pred, markets))
    if not args.json:
        print(DISCLAIMER)
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    fixtures = _load_fixtures(args.fixtures, args.sport)
    if not fixtures:
        raise SystemExit(f"Aucune rencontre dans {args.fixtures}")
    agent = _fit_football(args) if args.sport == "football" else _fit_esport(args)
    predictions = [agent.predict(fx) for fx in fixtures]

    if args.json:
        print(json.dumps(
            [json.loads(prediction_to_json(p, indent=None)) for p in predictions],
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"\n{BANNER}\nScan de {len(fixtures)} rencontre(s) — bankroll {args.bankroll:.2f} EUR, "
          f"edge minimum {args.min_edge:.1%}, Kelly x{args.kelly}\n")
    header = f"{'Rencontre':<34}{'Marche':<14}{'Selection':<16}{'Cote':>6}{'Edge':>8}{'Mise':>9}"
    print(header)
    print("-" * len(header))
    total = 0.0
    picks = 0
    for pred in predictions:
        for s in pred.selections:
            if not s.stake:
                continue
            picks += 1
            total += s.stake
            print(
                f"{pred.fixture.label[:33]:<34}{s.market:<14}{str(s.pick)[:15]:<16}"
                f"{s.odds:>6.2f}{s.edge:>+8.1%}{s.stake:>9.2f}"
            )
    if not picks:
        print("Aucun pari a valeur sur ce lot : le marche est efficient, on ne force pas.")
    else:
        print("-" * len(header))
        print(f"{picks} pari(s) — engagement total {total:.2f} EUR "
              f"({total / args.bankroll:.1%} de bankroll)")
    if args.detail:
        for pred in predictions:
            if any(s.stake for s in pred.selections):
                print(render_prediction(pred))
    print()
    print(DISCLAIMER)
    return 0


def cmd_ratings(args: argparse.Namespace) -> int:
    if args.sport == "football":
        agent = _fit_football(args)
        rows = [
            (t, agent.elo.rating(t), agent.model.strength(t))
            for t in sorted(agent.model.attack, key=lambda x: -agent.elo.rating(x))
        ]
        print(f"\n{'Equipe':<22}{'Elo':>7}{'Attaque':>10}{'Defense':>10}")
        print("-" * 49)
        for team, elo, st in rows:
            print(f"{team:<22}{elo:>7.0f}{st['attaque']:>10.2f}{st['defense']:>10.2f}")
        print(
            f"\nAvantage du terrain estime : "
            f"{(math.exp(agent.model.home_advantage) - 1) * 100:.1f}% de buts en plus. "
            f"Correction petits scores (rho) : {agent.model.rho:+.2f}. "
            f"{agent.model.n_matches} matchs."
        )
    else:
        agent = _fit_esport(args)
        print(f"\n{'Equipe':<22}{'Elo':>7}{'Maps':>7}")
        print("-" * 36)
        for team, elo, games in agent.elo.leaderboard(args.top):
            print(f"{team:<22}{elo:>7.0f}{games:>7}")
        print(f"\n{len(agent.results)} maps analysees.")
    print()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    matches = load_matches_csv(args.results)
    res = backtest_football(
        matches,
        train_min=args.train_min,
        refit_every=args.refit_every,
        config=_config(args),
        stake_flat=args.flat,
    )
    if args.json:
        print(json.dumps(
            {"resume": res.summary(), "calibration": res.calibration, "par_edge": res.by_edge},
            indent=2, ensure_ascii=False,
        ))
        return 0
    s = res.summary()
    print(f"\nBacktest walk-forward — {s['matchs_evalues']} matchs evalues\n")
    print(f"  Log-loss modele  : {s['log_loss_modele']}")
    print(f"  Log-loss marche  : {s['log_loss_marche']}  (cotes de cloture devigees)")
    print(f"  Gain vs marche   : {s['gain_vs_marche']:+.4f}  "
          f"({'le modele bat la cloture' if s['gain_vs_marche'] > 0 else 'le marche reste devant'})")
    print(f"  Brier (3 issues) : {s['brier_multiclasse']}")
    print(f"\n  Paris joues      : {s['paris']}  |  mise totale {s['mise_totale']} EUR")
    print(f"  P&L              : {s['pnl']:+.2f} EUR  |  ROI {s['roi_pct']:+.2f}%")
    print(f"  Taux de reussite : {s['taux_reussite_pct']}%  |  drawdown max {s['drawdown_max_pct']}%")
    if res.calibration:
        print("\n  Calibration (proba annoncee vs frequence reelle)")
        for row in res.calibration:
            print(f"    {row['tranche']:>9} : annonce {row['proba_moyenne']:.3f} — "
                  f"observe {row['frequence_reelle']:.3f}  (n={row['n']})")
    if res.by_edge:
        print("\n  ROI par tranche d'edge")
        for k, v in res.by_edge.items():
            print(f"    {k:<12} : {v['paris']:>4} paris, ROI {v['roi_pct']:+.2f}%")
    print(
        "\n  Lecture : un backtest positif sur des cotes de cloture est rare et fragile. "
        "\n  S'il l'est sur des cotes d'ouverture uniquement, c'est un signe de sur-apprentissage."
    )
    print()
    return 0


def cmd_bankroll(args: argparse.Namespace) -> int:
    bk = Bankroll(path=args.file, starting=args.starting)
    if args.action == "status":
        print(json.dumps(bk.stats(), indent=2, ensure_ascii=False))
    elif args.action == "list":
        print(summarise(bk.bets))
    elif args.action == "add":
        bet = bk.add(
            sport=args.sport, event=args.event, market=args.market, pick=args.pick,
            odds=args.odds_value, stake=args.stake, model_prob=args.prob, edge=args.edge,
            note=args.note,
        )
        print(f"Pari enregistre : {bet.id} — {bet.event} / {bet.pick} @ {bet.odds} "
              f"pour {bet.stake} EUR")
    elif args.action == "settle":
        bet = bk.settle(args.bet_id, args.status, closing_odds=args.closing)
        clv = bet.clv
        print(f"Pari {bet.id} regle : {bet.status}, P&L {bet.pnl:+.2f} EUR"
              + (f", CLV {clv:+.2f}%" if clv is not None else ""))
    return 0


def cmd_odds(args: argparse.Namespace) -> int:
    from .providers import TheOddsAPI

    api = TheOddsAPI(api_key=args.api_key)
    if args.list_sports:
        for s in api.sports():
            print(f"{s.get('key', ''):<40}{s.get('title', '')}")
        return 0
    fixtures = api.odds(args.sport_key, regions=args.regions, markets=args.markets)
    save_fixtures_json(args.out, fixtures)
    print(f"{len(fixtures)} rencontre(s) ecrite(s) dans {args.out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print(BANNER)
    print("\n--- 1. Forces des equipes (donnees de demo, 100% synthetiques) ---")
    ns = argparse.Namespace(**{**vars(args), "sport": "football", "results": DEMO_FOOT, "top": 20})
    cmd_ratings(ns)
    print("--- 2. Scan des rencontres a venir (football) ---")
    cmd_value(argparse.Namespace(**{**vars(ns), "fixtures": DEMO_FIXTURES, "detail": False, "json": False}))
    print("--- 3. Scan des rencontres a venir (esport, Bo3) ---")
    cmd_value(argparse.Namespace(**{
        **vars(ns), "sport": "esport", "results": DEMO_ESPORT,
        "fixtures": DEMO_ESPORT_FIXTURES, "detail": False, "json": False,
    }))
    print("--- 4. Backtest walk-forward sur l'historique ---")
    cmd_backtest(argparse.Namespace(**{
        **vars(ns), "results": DEMO_FOOT, "train_min": 260, "refit_every": 25,
        "flat": 10.0, "json": False,
    }))
    return 0


# --------------------------------------------------------------------------
# Parseur
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="betiq", description=BANNER)
    p.add_argument("--json", action="store_true", help="sortie JSON brute")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bankroll", type=float, default=1000.0, help="bankroll en euros")
    common.add_argument("--kelly", type=float, default=0.25, help="fraction de Kelly (0.25 = quart)")
    common.add_argument("--max-stake", type=float, default=0.02, dest="max_stake",
                        help="mise maximale en fraction de bankroll")
    common.add_argument("--min-edge", type=float, default=0.03, dest="min_edge",
                        help="edge minimum pour miser (0.03 = 3%%)")
    common.add_argument("--min-odds", type=float, default=1.30, dest="min_odds")
    common.add_argument("--max-odds", type=float, default=10.0, dest="max_odds")
    common.add_argument("--weight-model", type=float, default=0.45, dest="weight_model",
                        help="poids du modele face au marche (0-1)")
    common.add_argument("--devig", default="shin", choices=["shin", "power", "multiplicative"])

    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("predict", parents=[common], help="pronostic d'une rencontre")
    pr.add_argument("--sport", default="football", choices=["football", "esport"])
    pr.add_argument("--results", default=None, help="CSV de resultats passes")
    pr.add_argument("--home", required=True, help="equipe a domicile / equipe A")
    pr.add_argument("--away", required=True, help="equipe a l'exterieur / equipe B")
    pr.add_argument("--league", default="")
    pr.add_argument("--best-of", type=int, default=3, dest="best_of", help="esport : Bo1/Bo3/Bo5")
    pr.add_argument("--neutral", action="store_true", help="terrain neutre")
    pr.add_argument("--odds", default=None, help='cotes, ex. "1=2.10,X=3.40,2=3.60"')
    pr.add_argument("--odds-ou", default=None, dest="odds_ou",
                    help='cotes total buts, ex. "+2.5=1.90,-2.5=1.95"')
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_predict)

    va = sub.add_parser("value", parents=[common], help="scanner un lot de rencontres")
    va.add_argument("--sport", default="football", choices=["football", "esport"])
    va.add_argument("--results", default=None)
    va.add_argument("--fixtures", default=None, help="CSV ou JSON des rencontres a venir")
    va.add_argument("--detail", action="store_true", help="afficher l'analyse complete")
    va.add_argument("--json", action="store_true")
    va.set_defaults(func=cmd_value)

    ra = sub.add_parser("ratings", parents=[common], help="classement des forces d'equipes")
    ra.add_argument("--sport", default="football", choices=["football", "esport"])
    ra.add_argument("--results", default=None)
    ra.add_argument("--top", type=int, default=25)
    ra.add_argument("--json", action="store_true")
    ra.set_defaults(func=cmd_ratings)

    bt = sub.add_parser("backtest", parents=[common], help="backtest walk-forward (football)")
    bt.add_argument("--results", default=None)
    bt.add_argument("--train-min", type=int, default=260, dest="train_min")
    bt.add_argument("--refit-every", type=int, default=25, dest="refit_every")
    bt.add_argument("--flat", type=float, default=None,
                    help="mise fixe en euros au lieu du Kelly")
    bt.add_argument("--json", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    bk = sub.add_parser("bankroll", help="carnet de paris et suivi de bankroll")
    bk.add_argument("action", choices=["status", "list", "add", "settle"])
    bk.add_argument("--file", default=os.path.expanduser("~/.betiq/bankroll.json"))
    bk.add_argument("--starting", type=float, default=1000.0)
    bk.add_argument("--sport", default="football")
    bk.add_argument("--event", default="")
    bk.add_argument("--market", default="1X2")
    bk.add_argument("--pick", default="")
    bk.add_argument("--odds-value", type=float, default=0.0, dest="odds_value")
    bk.add_argument("--stake", type=float, default=0.0)
    bk.add_argument("--prob", type=float, default=0.0)
    bk.add_argument("--edge", type=float, default=0.0)
    bk.add_argument("--note", default="")
    bk.add_argument("--bet-id", default="", dest="bet_id")
    bk.add_argument("--status", default="won",
                    choices=["won", "lost", "void", "half_won", "half_lost"])
    bk.add_argument("--closing", type=float, default=None, help="cote de cloture (CLV)")
    bk.add_argument("--json", action="store_true")
    bk.set_defaults(func=cmd_bankroll)

    od = sub.add_parser("odds", help="recuperer les cotes en direct (the-odds-api.com)")
    od.add_argument("--api-key", default=None, dest="api_key")
    od.add_argument("--sport-key", default="soccer_france_ligue_one", dest="sport_key")
    od.add_argument("--regions", default="eu")
    od.add_argument("--markets", default="h2h,totals")
    od.add_argument("--out", default="fixtures.json")
    od.add_argument("--list-sports", action="store_true", dest="list_sports")
    od.add_argument("--json", action="store_true")
    od.set_defaults(func=cmd_odds)

    dm = sub.add_parser("demo", parents=[common], help="demonstration complete hors ligne")
    dm.add_argument("--json", action="store_true")
    dm.set_defaults(func=cmd_demo)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Valeurs par defaut : le jeu de demo, pour que tout marche sans configuration.
    if getattr(args, "results", None) is None:
        args.results = DEMO_ESPORT if getattr(args, "sport", "") == "esport" else DEMO_FOOT
    if getattr(args, "fixtures", None) is None and args.command == "value":
        args.fixtures = (
            DEMO_ESPORT_FIXTURES if getattr(args, "sport", "") == "esport" else DEMO_FIXTURES
        )
    for attr in ("results", "fixtures"):
        path = getattr(args, attr, None)
        if path and not os.path.exists(path):
            raise SystemExit(
                f"Fichier introuvable : {path}\n"
                "Passez --results / --fixtures, ou regenerez le jeu de demo avec "
                "`python3 scripts/generate_demo_data.py`."
            )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
