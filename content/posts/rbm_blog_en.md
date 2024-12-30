---
title : 'Energy based Model'
date : 2024-12-28T14:18:00+01:00
draft : false
showReadingTime : true
showPostNavLinks : true
---

# Energy based model
Les modèles classiques de machine learning reposent souvent sur des hypothèses sur la forme des données.
Par exemple les modèles type VAE forcent la distribution encodeur à s'approcher d'une gausienne.

L'idée générale est, une fois une distribution hypothèse fixée, d'ajuster différents paramètres jusqu'à obtenir une distribution optimale selon une fonction objectif ou de calculer les paramètres directement à partir des données lorsque cela est possible.

Ces méthodes sont performantes mais peuvent avoir des coûts de calculs importants dans des espaces à grandes dimensions ou sont directement limitées par les données à disposition.

Une idée est de sintéresser à comment la nature fonctionne.
Prenons l'exemple d'une balle lancée sur Terre.

![](image/energy/pomme.png)
Afin d'inférer la trajectoire $y$ de la pomme à partir des informations initiales $x$, la physique ne nous force pas à utiliser des algorithmes nécessitant de parcourir l'espace des trajectoires possibles.
Les lois de la physique choisissent simplement la courbe qui minimise l'énergie mécanique totale.
L'expression de cette énergie est parfaitement connue à chaque instant $$\mathcal{E} = \frac 1 2 .mv^2 + mgy$$ et donc la forme de la trajectoire optimale en découle directement.
![alt text](image/energy/ep.png)

On a donc mis en évidence le fait que l'inférence de $y$ grâce à $x$ passe par une minimisation d'une fonction qui a l'avantage de ne pas nécessiter d'approximation sur la forme des données, ni même de requerir un label de $x$.

Pour appliquer cette idée au machine learning il faudrait disposer d'une fonction "énergie" qu'il suffirait de minimiser afin d'inférer une sortie $y$ connaissant une entrée $x$.

Chaque problème de machine learning évolue dans son propre espace qui est induit par les données à disposition.
Ainsi la tâche d'entrainement ne serait pas de trouver des poids optimaux à l'aide d'une fonction de perte mais de déterminer une fonction énergie optimale pour le problème.

Cet article présentera différentes solutions de fonction énergie et les méthodes pour les optimiser.

## Hopfiled Networks
Le précurseur des modèles energy-based sont les ``Hopfield Networks``.
L'idée est d'avoir un système dit "dynamique" afin d'inférer une donnée $y$ à l'aide d'une donnée $x$.

Le cas d'application principal est la reconstruction de données binaires.
Par exemple : je donne une image binaire de A légérement modifiée en entrée.
Le modèle doit être capable de "réparer" l'image est de donner un A complet en sortie.

Ce genre de modèle est un exemple de mémoire associative.
Il retient des motifs pour être capable de reconnaitre des motifs même incomplet (à la manière d'un cerveau humain).

Posons le cadre mathématiques.
On considère des images binaires $\{x^0, x^1, \text{...}, x^p\}$ chacune de taille $N \times N$.

L'idée est d'utiliser un graphe non-orienté à $N$ noeuds pondérés par $W$ de taille $N \times N$ pour représenter les données.
Chaque noeud $i$ porte une valeur $x_i$ à $+1$ ou $-1$ (représentant la valeur du pixel binaire).

Afin de représenter la capacité qu'ont deux noeuds à être identique ou non on considère le produit x_i.x_j. 
On construit la matrice de poids $W$ où $w_{i,j}= \frac 1 P \sum_{k=0}^Px^k_i x^k_j$.
On interprète le poids $w_{i,j}$ comme la mesure moyenne de la capacité qu'ont deux noeuds à être identiques.

Si $w_{i,j} \gt 0$ alors les noeuds $i$ et $j$ ont tendance à être identiques.

L'idée est d'être capable de retrouver un motif retenu en donnant un motif incomplet au réseau.
On considère que le réseau de Hopfield a retenu une partie des $P$ motifs.

On condière une nouvelle entrée $x^{p+1}$. On charge les $x_i^{p+1}$ sur le graphe.
L'objectif est d'ajuster $x^{p+1}$ de manière à retrouver le motif le plus proche.

Considérons une connexion entre $x_i^{p+1}$ et $x_j^{p+1}$. 
Le poids $w_{i,j}$ est connu.

Si $w_{i,j} \gt 0$ on peut s'attendre à ce que $x_i^{p+1}$ et $x_j^{p+1}$ soient de même signe.
On considère donc le produit $w_{i,j}.x_i^{p+1}.x_j^{p+1}$.
L'objectif est donc que ce produit soit positif (et le plus grand possible).

On ajuste alors les différentes valeurs de $x^{p+1}$ itérativement de manière à maximiser ce produit.

Le calcul du nouvel état du neuronne se fait par 
$$x_i(t+1) =
\begin{cases} 
1 & \text{si } \sum_j w_{ij} x_j > 0, \\
-1 & \text{sinon.}
\end{cases}$$

Si on génèralise cette contraine à tout le réseau on obtient que résoudre notre problème revient à maximiser $\sum_{i=0}^P w_{i,j}.x_i x_j$.
Il suffit donc de minimser cette fonction avec un $-$.
La fonction énergie pour ce modèle est : $$\mathcal E(x) = -\sum_{i=0}^P w_{i,j}.x_i x_j $$

Le parallèle avec l'énergie en physique est fort. Les motifs mémorisés sont des minima locaux de cette fonction $\mathcal E$.
Lorsqu'on passe une entrée $x$, elle descend la courbe d'énergie jusqu'au minimum local le plus proche. Le système atteint alors un équilibre : il s'agit de la sortie $y$, le motif complet prédit.

Les Hopfield Networks ont pour limite que pour $N$ noeuds, ils ne retiennent entre $0.14N$ en $0.15N$ motifs.
Au delà, des états stables (minima d'energie $\mathcal E$) ne corespondant à aucun motif apparaissent.

## Différence avec une fonction de perte classique
Maintenant que nous avons vu une premier exemple de modèle basé sur une fonction énérgie une question essentielle peut se poser.
**L'energie n'est-elle alors pas simplement une fonction de perte ?**

La différence apportée par l'énergie est que l'énergie est utilisée à la fois à l'entrainement et à l'inférence.
Pendant la phase d'entrainement la fonction d'énérgie est construite.
Pendant l'inférence elle est utilisée puisqu'elle tente d'atteindre un état final avec énergie minimale.
Le modèle est dit dynamique : il évolue même pendant l'inférence (là où un réseau de neuronne supervisé classique est figé une fois ses poids appris).

Par ailleurs l'énergie ne mesure pas une différentre la sortie prédite $\hat y$ et un label $y$.
Ces fonctions fonctionnent également en non-supervisé puisque l'idée est simplement de donner au système un état stable (ce qui correspond à une énergie minimale).

## Un modèle basé énergie génératif : Restricted Boltzmann Machine (RBM)
