"""Tests unitaires de BetIQ (stdlib uniquement : python3 -m unittest discover tests)."""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betiq.bankroll import Bankroll
from betiq.engine import AgentConfig, EsportsAgent, FootballAgent
from betiq.markets.esports import series_score_probs, series_win_prob
from betiq.markets.football import football_markets
from betiq.models import Fixture, MapResult, Match, Odds
from betiq.providers.csv_source import load_fixtures_csv, load_maps_csv, load_matches_csv
from betiq.ratings.elo import EloConfig, EloRating
from betiq.ratings.poisson import DixonColes, DixonColesConfig
from betiq.value import StakingPlan, blend, devig, edge, kelly_fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_FOOT = os.path.join(ROOT, "data", "demo", "football_resultats.csv")
DEMO_ESPORT = os.path.join(ROOT, "data", "demo", "esport_maps.csv")


def _poisson(lam: float, rng: random.Random) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def synthetic_matches(n: int = 700, seed: int = 7) -> list[Match]:
    rng = random.Random(seed)
    strength = {"A": 1.6, "B": 1.35, "C": 1.1, "D": 0.95, "E": 0.8, "F": 0.65}
    teams = list(strength)
    out = []
    d0 = date(2025, 1, 1)
    for k in range(n):
        h, a = rng.sample(teams, 2)
        lam = 1.35 * strength[h] / strength[a]
        mu = 1.05 * strength[a] / strength[h]
        out.append(
            Match(d0 + timedelta(days=k // 3), h, a, _poisson(lam, rng), _poisson(mu, rng))
        )
    return out


class TestDevig(unittest.TestCase):
    prices = {"1": 2.10, "X": 3.40, "2": 3.60}

    def test_sums_to_one(self):
        for method in ("multiplicative", "power", "shin"):
            probs = devig(self.prices, method)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6, msg=method)

    def test_shin_reduces_longshot(self):
        mult = devig(self.prices, "multiplicative")
        shin = devig(self.prices, "shin")
        # Shin corrige le biais favori/outsider : l'outsider perd de la proba.
        self.assertLess(shin["2"], mult["2"])
        self.assertGreater(shin["1"], mult["1"])

    def test_fair_book_untouched(self):
        fair = {"A": 2.0, "B": 2.0}
        self.assertAlmostEqual(devig(fair, "shin")["A"], 0.5, places=6)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            devig(self.prices, "magique")


class TestValue(unittest.TestCase):
    def test_edge_and_kelly(self):
        self.assertAlmostEqual(edge(0.5, 2.0), 0.0, places=9)
        self.assertAlmostEqual(kelly_fraction(0.5, 2.0), 0.0, places=9)
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.2, places=9)
        self.assertEqual(kelly_fraction(0.4, 2.0), 0.0)  # jamais de mise negative

    def test_blend_between_bounds(self):
        b = blend(0.70, 0.40, 0.5)
        self.assertLess(b, 0.70)
        self.assertGreater(b, 0.40)
        self.assertAlmostEqual(blend(0.7, 0.4, 1.0), 0.7, places=6)
        self.assertAlmostEqual(blend(0.7, 0.4, 0.0), 0.4, places=6)

    def test_staking_caps_and_filters(self):
        plan = StakingPlan(bankroll=1000, kelly_fraction_used=0.25, max_stake_pct=0.02,
                           min_edge=0.03, min_odds=1.3, max_odds=10.0)
        _, stake = plan.stake(0.99, 5.0)          # Kelly enorme -> plafonne a 2%
        self.assertEqual(stake, 20.0)
        self.assertEqual(plan.stake(0.51, 2.0)[1], 0.0)   # edge sous le seuil
        self.assertEqual(plan.stake(0.99, 1.05)[1], 0.0)  # cote hors plage


class TestFootballMarkets(unittest.TestCase):
    def setUp(self):
        model = DixonColes()
        model.attack = {"A": 0.3, "B": -0.1}
        model.defence = {"A": 0.1, "B": -0.2}
        model.base, model.home_advantage, model.rho = 0.1, 0.25, -0.05
        self.markets = football_markets(model.score_matrix("A", "B"))

    def test_1x2_is_a_distribution(self):
        self.assertAlmostEqual(sum(self.markets["1X2"].values()), 1.0, places=6)

    def test_double_chance_consistent(self):
        o = self.markets["1X2"]
        self.assertAlmostEqual(self.markets["double_chance"]["1X"], o["1"] + o["X"], places=9)

    def test_totals_complement(self):
        t = self.markets["totals"]
        self.assertAlmostEqual(t["+2.5"] + t["-2.5"], 1.0, places=9)

    def test_asian_handicap_mirror(self):
        ah = self.markets["ah"]
        for line in ("0.5", "1", "0.25"):
            self.assertAlmostEqual(ah[f"dom -{line}"] + ah[f"ext +{line}"], 1.0, places=6)
        self.assertAlmostEqual(ah["dom +0"] + ah["ext +0"], 1.0, places=6)

    def test_handicap_ordering(self):
        ah = self.markets["ah"]
        self.assertGreater(ah["dom +0.5"], ah["dom -0.5"])  # ligne plus favorable = plus probable


class TestEsportsMarkets(unittest.TestCase):
    def test_series_math(self):
        self.assertAlmostEqual(series_win_prob(0.6, 1), 0.6, places=9)
        self.assertAlmostEqual(series_win_prob(0.6, 3), 0.648, places=6)
        self.assertAlmostEqual(series_win_prob(0.6, 5), 0.68256, places=5)

    def test_format_amplifies_favourite(self):
        for p in (0.55, 0.62, 0.75):
            self.assertLess(series_win_prob(p, 1), series_win_prob(p, 3))
            self.assertLess(series_win_prob(p, 3), series_win_prob(p, 5))

    def test_coin_flip_stays_even(self):
        for bo in (1, 3, 5):
            self.assertAlmostEqual(series_win_prob(0.5, bo), 0.5, places=6)

    def test_scores_sum_to_one(self):
        for bo in (1, 3, 5):
            self.assertAlmostEqual(sum(series_score_probs(0.62, bo).values()), 1.0, places=9)


class TestElo(unittest.TestCase):
    def test_update_direction(self):
        elo = EloRating(EloConfig(k=20, home_advantage=0))
        elo.update("A", "B", 1.0)
        self.assertGreater(elo.rating("A"), 1500)
        self.assertLess(elo.rating("B"), 1500)
        self.assertAlmostEqual(elo.rating("A") + elo.rating("B"), 3000, places=6)

    def test_home_advantage_raises_expectation(self):
        elo = EloRating(EloConfig(home_advantage=60))
        self.assertGreater(elo.expected("A", "B"), 0.5)
        self.assertAlmostEqual(elo.expected("A", "B", neutral=True), 0.5, places=9)

    def test_roundtrip(self):
        elo = EloRating(EloConfig(k=25))
        elo.update("A", "B", 1.0)
        clone = EloRating.from_dict(elo.to_dict())
        self.assertEqual(clone.ratings, elo.ratings)
        self.assertEqual(clone.config.k, 25)


class TestDixonColes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = synthetic_matches()
        cls.model = DixonColes(DixonColesConfig(half_life_days=3000, iterations=600)).fit(cls.matches)

    def test_recovers_ranking(self):
        order = sorted(self.model.attack, key=lambda t: -self.model.attack[t])
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "F")

    def test_home_advantage_positive(self):
        # Le generateur donne 1.35 buts a domicile contre 1.05 a l'exterieur.
        self.assertGreater(self.model.home_advantage, 0.1)
        self.assertLess(self.model.home_advantage, 0.4)

    def test_expected_goals_ordering(self):
        strong, _ = self.model.expected_goals("A", "F")
        weak, _ = self.model.expected_goals("F", "A")
        self.assertGreater(strong, weak)

    def test_score_matrix_normalised(self):
        grid = self.model.score_matrix("A", "B")
        self.assertAlmostEqual(sum(sum(r) for r in grid), 1.0, places=6)

    def test_roundtrip(self):
        clone = DixonColes.from_dict(self.model.to_dict())
        self.assertAlmostEqual(clone.expected_goals("A", "B")[0],
                               self.model.expected_goals("A", "B")[0], places=9)


class TestAgents(unittest.TestCase):
    def test_football_agent_finds_value_on_inflated_odds(self):
        agent = FootballAgent(config=AgentConfig(weight_model=0.6)).fit(synthetic_matches())
        probs = football_markets(agent.model.score_matrix("A", "F"))["1X2"]
        generous = round(1 / probs["1"] * 1.6, 2)  # cote tres au-dessus du juste prix
        fx = Fixture(home="A", away="F", odds=[Odds("1X2", {"1": generous, "X": 4.0, "2": 9.0})])
        pred = agent.predict(fx)
        best = pred.best_bet
        self.assertIsNotNone(best)
        self.assertEqual(best.pick, "1")
        self.assertGreater(best.edge, 0.03)
        self.assertGreater(best.stake, 0)

    def test_football_agent_passes_on_fair_odds(self):
        agent = FootballAgent().fit(synthetic_matches())
        probs = football_markets(agent.model.score_matrix("A", "F"))["1X2"]
        fair = {k: round(1 / v / 1.06, 2) for k, v in probs.items()}  # marge de 6%
        fx = Fixture(home="A", away="F", odds=[Odds("1X2", fair)])
        pred = agent.predict(fx)
        self.assertIsNone(pred.best_bet)  # aucune valeur : on ne mise pas

    def test_esports_agent_series_probability(self):
        rng = random.Random(3)
        truth = {"Alpha": 1700, "Beta": 1500, "Gamma": 1400}
        maps = []
        d0 = date(2026, 1, 1)
        for k in range(600):
            a, b = rng.sample(list(truth), 2)
            p = 1 / (1 + 10 ** (-(truth[a] - truth[b]) / 400))
            w, l = (a, b) if rng.random() < p else (b, a)
            maps.append(MapResult(d0 + timedelta(days=k // 4), w, l))
        agent = EsportsAgent().fit(maps)
        self.assertGreater(agent.elo.rating("Alpha"), agent.elo.rating("Gamma"))
        bo1 = agent.predict(Fixture(home="Alpha", away="Gamma", sport="esport", best_of=1))
        bo5 = agent.predict(Fixture(home="Alpha", away="Gamma", sport="esport", best_of=5))
        self.assertLess(
            bo1.probabilities["vainqueur"]["Alpha"],
            bo5.probabilities["vainqueur"]["Alpha"],
        )

    def test_unknown_team_is_flagged(self):
        agent = FootballAgent().fit(synthetic_matches())
        pred = agent.predict(Fixture(home="A", away="Equipe Inconnue"))
        self.assertTrue(any("ATTENTION" in n for n in pred.notes))


class TestBankroll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.bk = Bankroll(path=self.tmp.name, starting=1000.0)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_pnl_and_clv(self):
        won = self.bk.add("football", "A vs B", "1X2", "1", 2.5, 20, 0.5, 0.25)
        self.bk.settle(won.id, "won", closing_odds=2.2)
        lost = self.bk.add("esport", "C vs D", "vainqueur", "C", 1.8, 10)
        self.bk.settle(lost.id, "lost")
        self.assertAlmostEqual(self.bk.pnl, 20.0, places=2)   # +30 -10
        self.assertAlmostEqual(self.bk.current, 1020.0, places=2)
        self.assertAlmostEqual(self.bk.get(won.id).clv, (2.5 / 2.2 - 1) * 100, places=6)

    def test_half_won_and_void(self):
        half = self.bk.add("football", "A vs B", "ah", "dom -0.25", 2.0, 100)
        self.bk.settle(half.id, "half_won")
        self.assertAlmostEqual(self.bk.get(half.id).pnl, 50.0, places=2)
        void = self.bk.add("football", "A vs B", "ah", "dom +0", 2.0, 100)
        self.bk.settle(void.id, "void")
        self.assertAlmostEqual(self.bk.get(void.id).pnl, 0.0, places=2)

    def test_persistence(self):
        bet = self.bk.add("football", "A vs B", "1X2", "1", 2.0, 10)
        reloaded = Bankroll(path=self.tmp.name)
        self.assertEqual(len(reloaded.bets), 1)
        self.assertEqual(reloaded.bets[0].id, bet.id)

    def test_pending_engages_bankroll(self):
        self.bk.add("football", "A vs B", "1X2", "1", 2.0, 50)
        self.assertAlmostEqual(self.bk.current, 950.0, places=2)

    def test_bad_status(self):
        bet = self.bk.add("football", "A vs B", "1X2", "1", 2.0, 10)
        with self.assertRaises(ValueError):
            self.bk.settle(bet.id, "gagne")


class TestProviders(unittest.TestCase):
    def test_demo_football_csv(self):
        matches = load_matches_csv(DEMO_FOOT)
        self.assertGreater(len(matches), 500)
        self.assertTrue(all(m.closing_odds for m in matches))
        self.assertEqual(matches, sorted(matches, key=lambda m: m.date))

    def test_demo_esport_csv(self):
        maps = load_maps_csv(DEMO_ESPORT)
        self.assertGreater(len(maps), 500)
        self.assertTrue(all(r.winner and r.loser for r in maps))

    def test_series_csv_is_expanded_into_maps(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write("date,team_a,team_b,maps_a,maps_b\n2026-01-01,Alpha,Beta,2,1\n")
            path = fh.name
        try:
            maps = load_maps_csv(path)
            self.assertEqual(len(maps), 3)
            self.assertEqual(sum(1 for m in maps if m.winner == "Alpha"), 2)
        finally:
            os.unlink(path)

    def test_fixtures_csv_odds(self):
        fixtures = load_fixtures_csv(os.path.join(ROOT, "data", "demo", "football_a_venir.csv"))
        self.assertTrue(fixtures)
        markets = {o.market for o in fixtures[0].odds}
        self.assertIn("1X2", markets)
        self.assertIn("totals", markets)


class TestCli(unittest.TestCase):
    def test_commands_run(self):
        from betiq.cli import main
        import contextlib
        import io

        for argv in (
            ["ratings", "--sport", "football"],
            ["ratings", "--sport", "esport"],
            ["value", "--sport", "football"],
            ["value", "--sport", "esport"],
            ["predict", "--home", "Rouen FC", "--away", "Verdun AC",
             "--odds", "1=1.50,X=4.20,2=6.00"],
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(argv)
            self.assertEqual(code, 0, argv)
            self.assertTrue(buf.getvalue().strip(), argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
