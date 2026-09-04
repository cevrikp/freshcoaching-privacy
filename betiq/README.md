# BetIQ — agent de pronostics sportifs et esport

Un agent en ligne de commande qui estime les probabilites d'un match a partir des
statistiques, les compare aux cotes du bookmaker, et ne conseille un pari que
lorsque l'ecart (l'*edge*) est suffisant. Il explique chaque pronostic, calcule la
mise selon un Kelly fractionne, et tient le carnet de paris.

**Football** (Dixon-Coles) et **esport** (Elo par map + conversion Bo1/Bo3/Bo5).

Aucune dependance : Python 3.10+ et la bibliotheque standard suffisent.

---

## Demarrage en 30 secondes

```bash
python3 -m betiq.cli demo          # demonstration complete, hors ligne
# ou, apres `pip install -e .` :
betiq demo
```

La demo tourne sur un jeu de donnees **entierement synthetique** (`data/demo/`) :
equipes fictives, cotes simulees. Il sert a valider le fonctionnement, pas a
prouver une rentabilite.

## Les cinq commandes

```bash
# 1. Forces des equipes (attaque / defense / Elo)
betiq ratings --sport football --results mes_resultats.csv

# 2. Pronostic d'une rencontre, avec ou sans cotes
betiq predict --home "Lyon" --away "Nice" --odds "1=2.05,X=3.50,2=3.70"
betiq predict --sport esport --home "G2" --away "Fnatic" --best-of 3 \
              --odds "G2=1.70,Fnatic=2.15"

# 3. Scanner un lot de rencontres et sortir les paris a valeur
betiq value --fixtures journee.csv --bankroll 500 --min-edge 0.04 --detail

# 4. Backtest walk-forward (aucune donnee future n'entre dans le modele)
betiq backtest --results historique.csv --flat 10

# 5. Carnet de paris et bankroll
betiq bankroll add --event "Lyon vs Nice" --market 1X2 --pick 1 \
                   --odds-value 2.05 --stake 12 --prob 0.53 --edge 0.086
betiq bankroll settle --bet-id a1b2c3d4 --status won --closing 1.92
betiq bankroll status
```

Ajoutez `--json` a n'importe quelle commande pour une sortie machine.

## Utiliser vos propres donnees

### Football

Le format le plus simple :

```csv
date,home,away,home_score,away_score,league
2026-08-16,Lyon,Nice,2,1,Ligue 1
```

Les CSV de **football-data.co.uk** (gratuits, historiques + cotes de cloture
Pinnacle/Bet365) sont lus directement, sans conversion :

```bash
betiq backtest --results F1.csv        # F1.csv telecharge tel quel
```

### Esport

Map par map :

```csv
date,winner,loser,game,event,winner_score,loser_score
2026-08-16,G2,Fnatic,CS2,ESL,13,9
```

…ou par serie, l'agent eclate le score en maps :

```csv
date,team_a,team_b,maps_a,maps_b
2026-08-16,G2,Fnatic,2,1
```

### Rencontres a venir et cotes

CSV (`date,home,away,league,best_of,odds_1,odds_x,odds_2,ou_line,odds_over,odds_under`)
ou JSON multi-marches :

```json
{"fixtures": [{
  "home": "Lyon", "away": "Nice", "league": "Ligue 1",
  "odds": [
    {"market": "1X2", "bookmaker": "book", "prices": {"1": 2.05, "X": 3.5, "2": 3.7}},
    {"market": "totals", "prices": {"+2.5": 1.9, "-2.5": 1.95}}
  ]}]}
```

### Cotes en direct (optionnel)

```bash
export ODDS_API_KEY="votre_cle"        # the-odds-api.com, palier gratuit
betiq odds --list-sports
betiq odds --sport-key soccer_france_ligue_one --out journee.json
betiq value --fixtures journee.json --results F1.csv
```

## Ce que fait le modele

| Etape | Football | Esport |
|---|---|---|
| Force des equipes | Dixon-Coles : attaque + defense + avantage du terrain, ponderes par recence | Elo par map (K eleve : les rosters bougent vite) |
| Grille de resultats | Poisson bivarie avec correction des petits scores | Conversion probabilite de map → probabilite de serie |
| Marches derives | 1X2, double chance, over/under, BTTS, handicaps asiatiques (quarts inclus), score exact, totaux par equipe | Vainqueur, score de serie, handicap de maps, total de maps |
| Face au marche | retrait de la marge (Shin / power / multiplicative), puis melange modele + marche en espace logit | idem |
| Decision | edge ≥ seuil, cote dans la plage, Kelly fractionne plafonne | idem |

**Le melange avec le marche est volontaire.** Un bookmaker serieux integre les
compositions, les blessures et les paris des professionnels. Ignorer son prix,
c'est confondre "mon modele voit autre chose" avec "j'ai raison". Par defaut le
modele pese 45%, le marche 55% (`--weight-model`).

## Reglages utiles

| Option | Defaut | Effet |
|---|---|---|
| `--bankroll` | 1000 | capital de reference des mises |
| `--kelly` | 0.25 | fraction de Kelly (quart de Kelly = standard prudent) |
| `--max-stake` | 0.02 | plafond absolu par pari (2% de bankroll) |
| `--min-edge` | 0.03 | edge minimum pour miser |
| `--min-odds` / `--max-odds` | 1.30 / 10 | plage de cotes jouables |
| `--weight-model` | 0.45 | poids du modele face au marche |
| `--devig` | shin | methode de retrait de la marge |

## Tests

```bash
python3 -m unittest discover -s tests -v      # 38 tests, ~4 s
```

## Limites — a lire avant de miser

- Le modele ne connait **ni les blessures, ni les compositions, ni la motivation**
  (match sans enjeu, rotation avant une coupe). Verifiez toujours avant de valider.
- Sur un bookmaker serieux, un edge affiche sous 3-4% est le plus souvent du bruit
  de modele, pas de la valeur.
- Le seul indicateur exploitable a court terme est la **CLV** (avoir pris une cote
  meilleure que la cloture). Le ROI demande plusieurs centaines de paris pour
  vouloir dire quelque chose : c'est pour cela que `bankroll settle --closing`
  existe.
- Les bookmakers limitent les comptes gagnants. Un edge reel se heurte tot ou tard
  a un plafond de mise.
- Le backtest fourni tourne sur des cotes de **cloture** : c'est la barre la plus
  haute. Un backtest positif sur des cotes d'ouverture uniquement signale surtout
  du sur-apprentissage.

## Jeu responsable

Parier comporte un risque de perte. Aucun modele ne garantit un gain : l'objectif
est un avantage faible sur un grand nombre de paris, avec de longues series
perdantes en chemin. Ne misez que ce que vous pouvez perdre, jamais d'argent
emprunte, et n'essayez pas de "refaire" une perte.

**Aide et conseil : joueurs-info-service.fr — 09 74 75 13 13** (appel anonyme et
gratuit, 8h-2h, 7j/7).
