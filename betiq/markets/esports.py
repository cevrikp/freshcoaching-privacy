"""Marches esport : d'une probabilite par map vers la serie complete.

Les cotes esport portent souvent sur la serie (Bo3 / Bo5) alors que le
modele estime une probabilite par map. La conversion n'est pas lineaire :
un favori a 60% par map gagne un Bo3 a 64.8% et un Bo5 a 68.3%. C'est
exactement la ou le marche des petits tournois se trompe le plus souvent.
"""

from __future__ import annotations

from math import comb


def _clip(p: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, p))


def series_score_probs(p_map: float, best_of: int) -> dict[str, float]:
    """Probabilite de chaque score de serie, vu du camp A.

    Cle "2-1" = A gagne 2 maps a 1 ; "1-2" = A perd 1-2.
    """
    p = _clip(p_map)
    if best_of <= 1:
        return {"1-0": p, "0-1": 1.0 - p}
    need = best_of // 2 + 1
    out: dict[str, float] = {}
    for lost in range(need):
        out[f"{need}-{lost}"] = comb(need - 1 + lost, lost) * p**need * (1 - p) ** lost
        out[f"{lost}-{need}"] = comb(need - 1 + lost, lost) * (1 - p) ** need * p**lost
    return out


def series_win_prob(p_map: float, best_of: int) -> float:
    """Probabilite de gagner la serie a partir de la probabilite par map."""
    scores = series_score_probs(p_map, best_of)
    need = max(1, best_of // 2 + 1)
    return sum(v for k, v in scores.items() if int(k.split("-")[0]) == need)


def esports_markets(p_map: float, best_of: int) -> dict[str, dict[str, float]]:
    """Vainqueur, handicap de maps, total de maps, score exact de serie."""
    scores = series_score_probs(p_map, best_of)
    pa = series_win_prob(p_map, best_of)
    out: dict[str, dict[str, float]] = {
        "vainqueur": {"A": pa, "B": 1.0 - pa},
        "score_serie": {k: v for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
    }

    if best_of >= 3:
        need = best_of // 2 + 1
        # Handicap -1.5 map : gagner sans perdre une seule map (Bo3).
        clean_a = scores.get(f"{need}-0", 0.0)
        clean_b = scores.get(f"0-{need}", 0.0)
        out["handicap_maps"] = {
            "A -1.5": clean_a if best_of == 3 else sum(
                v for k, v in scores.items()
                if int(k.split("-")[0]) == need and int(k.split("-")[1]) <= need - 2
            ),
            "A +1.5": 1.0 - clean_b if best_of == 3 else 1.0 - sum(
                v for k, v in scores.items()
                if int(k.split("-")[1]) == need and int(k.split("-")[0]) <= need - 2
            ),
            "B -1.5": clean_b if best_of == 3 else sum(
                v for k, v in scores.items()
                if int(k.split("-")[1]) == need and int(k.split("-")[0]) <= need - 2
            ),
            "B +1.5": 1.0 - clean_a if best_of == 3 else 1.0 - sum(
                v for k, v in scores.items()
                if int(k.split("-")[0]) == need and int(k.split("-")[1]) <= need - 2
            ),
        }
        totals: dict[str, float] = {}
        for line in _map_lines(best_of):
            over = sum(
                v
                for k, v in scores.items()
                if sum(int(x) for x in k.split("-")) > line
            )
            totals[f"+{line:g}"] = over
            totals[f"-{line:g}"] = 1.0 - over
        out["total_maps"] = totals
    return out


def _map_lines(best_of: int) -> list[float]:
    need = best_of // 2 + 1
    return [x + 0.5 for x in range(need, best_of)]
