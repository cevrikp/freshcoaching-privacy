# La methode, en detail

Ce document explique ce que fait chaque brique de l'agent et *pourquoi*, pour que
vous puissiez juger des resultats au lieu de les subir.

---

## 1. Estimer une force d'equipe (football)

Le modele **Dixon-Coles** attribue a chaque equipe une force offensive `atk` et une
force defensive `def`, plus un avantage du terrain global `h` :

```
buts attendus domicile  lambda = exp(base + atk_dom - def_ext + h)
buts attendus exterieur mu     = exp(base + atk_ext - def_dom)
```

Trois raffinements comptent :

**Ponderation temporelle.** Chaque match passe est pondere par `exp(-ln2 * jours / demi_vie)`
avec une demi-vie de 180 jours : un match d'il y a six mois pese moitie moins qu'un
match d'hier. Sans cela, un modele reste bloque sur l'equipe de la saison derniere.

**Correction des petits scores.** Un Poisson independant sous-estime les 0-0 et
1-1 et surestime les 1-0 / 0-1 : les equipes ne marquent pas independamment l'une
de l'autre. Dixon-Coles corrige les quatre cases concernees par un parametre `rho`
ajuste par vraisemblance.

**Regularisation.** Les forces sont tirees vers zero (L2). Une equipe avec cinq
matchs joues ne merite pas une force extreme.

L'ajustement se fait par montee de gradient (Adam) sur la log-vraisemblance
ponderee — en Python pur, donc sans numpy.

Une fois `lambda` et `mu` connus, on construit la grille des scores exacts (0-0 a
12-12) et **tous** les marches en decoulent par simple sommation : 1X2, double
chance, over/under sur n'importe quelle ligne, BTTS, handicaps asiatiques (y
compris les quarts de but, joues moitie sur chaque demi-ligne), totaux par equipe,
score exact. C'est la force de l'approche : un seul modele, coherent entre marches.

## 2. Estimer une force d'equipe (esport)

Pas de match nul, et des rosters qui changent tous les trois mois : l'Elo par map
est mieux adapte qu'un modele de score.

- `K = 28` (contre 20 en football) : la forme recente pese plus lourd.
- Marge de victoire prise en compte de facon logarithmique (un 13-2 informe plus
  qu'un 13-11, sans faire exploser le rating).
- Terrain neutre par defaut.
- **Amortissement de 10%** vers 50% : l'Elo esport surestime systematiquement les
  favoris, parce qu'il ignore les changements de joueurs, les patchs et la meta.

Puis la conversion vers la serie, qui est la ou le marche se trompe le plus
souvent sur les petits tournois :

| Proba par map | Bo1 | Bo3 | Bo5 |
|---|---|---|---|
| 55% | 55.0% | 57.5% | 59.3% |
| 60% | 60.0% | 64.8% | 68.3% |
| 70% | 70.0% | 78.4% | 83.7% |

Le format amplifie l'avantage du favori. Un Bo5 punit l'upset ; un Bo1 le
recompense. Une cote de Bo3 calquee sur une intuition de Bo1 est mal price.

## 3. Retirer la marge du bookmaker (devig)

La somme des probabilites implicites d'un marche depasse 1 : c'est la marge. Trois
methodes sont proposees.

- **Multiplicative** : on divise par la somme. Simple, mais elle repartit la marge
  proportionnellement et surestime donc les outsiders.
- **Power** : on cherche `k` tel que la somme des `q_i^k` fasse 1.
- **Shin** (defaut) : modele ou une fraction des parieurs est informee. C'est la
  reference du milieu, et celle qui corrige le mieux le biais favori/outsider.

Sur `1 = 2.10 / X = 3.40 / 2 = 3.60` (marge 4.8%), Shin donne `45.9% / 27.9% / 26.3%`
la ou la methode multiplicative donne `45.4% / 28.1% / 26.5%` : l'outsider perd
0.2 point. Sur des cotes longues, l'ecart devient decisif.

## 4. Melanger modele et marche

Le point le plus important, et celui que les "systemes" de pronostics ignorent.

```
p_finale = sigmoid( w * logit(p_modele) + (1-w) * logit(p_marche) )
```

Le melange se fait en espace logit (plus stable aux extremes qu'une moyenne
arithmetique), avec `w = 0.45` par defaut. Autrement dit : **le marche pese plus
que notre modele**, et il faut un desaccord substantiel pour qu'un pari sorte.

Si la marge du bookmaker est anormalement elevee (> 12%), le prix est moins
informatif : le poids du modele monte, et l'agent signale l'operateur.

## 5. Decider et miser

- **Edge** = `p * cote - 1`, l'esperance par euro mise.
- **Kelly** = `(p * cote - 1) / (cote - 1)`, la fraction qui maximise la croissance
  logarithmique du capital.
- On applique un **quart de Kelly**, plafonne a **2% de bankroll**. Le Kelly plein
  est mathematiquement optimal *si* la probabilite est exacte — elle ne l'est
  jamais, et une erreur d'estimation en Kelly plein ruine un capital.
- Filtres : edge minimum 3%, cotes entre 1.30 et 10.

## 6. Verifier — la partie que personne ne fait

Le backtest est **walk-forward** : a l'etape `i`, le modele n'a vu que les matchs
`0..i-1`. Aucune information future ne fuit.

Trois familles de resultats :

**Qualite probabiliste.** Log-loss du modele contre log-loss des cotes de cloture
devigees. Si votre modele ne bat pas la cloture, il n'a pas d'edge — point final.
C'est le test le plus severe, et le plus honnete.

**Calibration.** Sur les matchs annonces a 60%, l'issue arrive-t-elle 60% du temps ?
Un modele peut trier correctement et rester mal calibre ; dans ce cas les edges
affiches sont faux.

**Economique.** ROI, taux de reussite, drawdown maximal, et ROI par tranche d'edge.
Ce dernier tableau est un detecteur de mensonge : si les "edges 20%+" perdent de
l'argent alors que les "edges 3-5%" en gagnent, votre modele ne mesure pas ce que
vous croyez.

Sur le jeu de demonstration, le modele finit **legerement derriere** la cloture
(gain de log-loss negatif, ROI proche de zero). C'est le resultat attendu quand les
cotes sont derivees des vraies probabilites : le marche efficient gagne. Un agent
qui vous annonce +30% de ROI en backtest sur des cotes de cloture a un bug ou un
biais de look-ahead.

## 7. La CLV, seul indicateur a court terme

Le ROI a besoin de plusieurs centaines de paris pour sortir du bruit. La **closing
line value** — avoir pris 2.10 sur une selection qui ferme a 1.95, soit +7.7% — se
mesure des le premier pari, et elle est fortement correlee a la rentabilite long
terme.

```bash
betiq bankroll settle --bet-id a1b2c3d4 --status won --closing 1.95
betiq bankroll status     # clv_moyenne_pct, clv_positive_pct
```

Si votre CLV moyenne est negative sur cinquante paris, votre modele n'a pas d'edge,
quel que soit votre solde du moment.

## 8. Ce que le modele ne voit pas

Aucun modele statistique ne connait :

- les compositions, blessures, suspensions, retours de blessure ;
- la motivation (maintien acquis, match sans enjeu, gestion avant une coupe) ;
- la meteo, l'etat du terrain, un deplacement long en milieu de semaine ;
- en esport : un changement de roster, un stand-in, un patch qui retourne la meta,
  un joueur malade.

Ce sont exactement les informations que le bookmaker, lui, integre. D'ou la regle :
l'agent propose une base chiffree et un ordre de grandeur de mise — la decision
finale reste la votre, apres verification des nouvelles du jour.

## 9. Jeu responsable

Parier comporte un risque de perte. Meme un edge reel s'accompagne de series
perdantes longues : avec un ROI de 3%, une serie de dix paris perdants d'affilee
est banale. Ne misez que ce que vous pouvez perdre, ne remontez jamais vos mises
pour "refaire" une perte, et fixez-vous une limite avant de commencer.

**joueurs-info-service.fr — 09 74 75 13 13** (anonyme et gratuit, 8h-2h, 7j/7).
