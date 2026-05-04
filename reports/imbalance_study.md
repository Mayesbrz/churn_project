# Étude comparative du déséquilibre des classes

## 1. Constat initial

- Classe 0 (non churn) : 8979 clients.
- Classe 1 (churn) : 1021 clients.
- Ratio de déséquilibre majorité/minorité : **8.79:1**.

Ce déséquilibre rend l'accuracy insuffisante : un modèle qui prédit majoritairement la classe 0 peut obtenir un bon score global tout en manquant les clients churn. Dans un contexte CRM, les faux négatifs sont coûteux car ils correspondent à des clients à risque non détectés.

## 2. Baseline et limites de l'accuracy

Baseline Logistic Regression au seuil 0.5 : accuracy 0.8965, recall 0.0098, F1 0.0190, PR-AUC 0.2392.
Matrice de confusion baseline : FP=5, FN=202.

Les métriques retenues sont donc : Recall, F1-score, ROC-AUC et PR-AUC. Le Recall mesure la capacité à détecter les churners, le F1 équilibre précision et rappel, ROC-AUC mesure la discrimination globale, et PR-AUC est particulièrement informative quand la classe positive est minoritaire.

## 3. Méthodes testées

- Baseline : aucun rééquilibrage.
- Class Weight : pondération automatique des classes dans les modèles compatibles.
- Random Over-Sampling : duplication aléatoire de la classe minoritaire.
- SMOTE : génération synthétique de churners.
- Random Under-Sampling : réduction de la classe majoritaire.
- Ajustement du seuil : recherche du seuil maximisant le F1 sur validation.

La validation utilise Stratified K-Fold afin de préserver les proportions de churn/non-churn dans chaque fold.

## 4. Meilleur modèle par stratégie

| Stratégie | Modèle | Precision | Recall | F1 | ROC-AUC | PR-AUC | Seuil | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Over-Sampling | Random Forest | 0.2629 | 0.7990 | 0.3956 | 0.8010 | 0.2679 | 0.390 | 457 | 41 |
| Random Under-Sampling | Random Forest | 0.2630 | 0.7696 | 0.3920 | 0.8046 | 0.3064 | 0.550 | 440 | 47 |
| Class Weight | Random Forest | 0.2478 | 0.8186 | 0.3804 | 0.7869 | 0.2640 | 0.310 | 507 | 37 |
| Baseline | Gradient Boosting | 0.2460 | 0.8235 | 0.3788 | 0.8092 | 0.3185 | 0.135 | 515 | 36 |
| SMOTE | Random Forest | 0.2458 | 0.7843 | 0.3743 | 0.7851 | 0.2630 | 0.235 | 491 | 44 |

## 5. Synthèse comparative

La meilleure configuration observée est **Random Over-Sampling + Random Forest** avec F1=0.3956, recall=0.7990, PR-AUC=0.2679 et seuil=0.390.

Effets observés :

- Le seuil 0.5 est rarement optimal dans ce contexte ; l'ajustement du seuil améliore la détection des churners.
- Random Over-Sampling et SMOTE augmentent généralement le rappel, avec un risque d'overfitting ou de bruit synthétique.
- Random Under-Sampling peut améliorer le rappel mais perd de l'information sur la classe majoritaire.
- Class Weight est simple, robuste et facile à déployer, mais ne suffit pas toujours à maximiser la détection.

## 6. Recommandation métier

La stratégie finale doit privilégier un compromis F1/Recall plutôt que l'accuracy. Dans un service CRM, il est acceptable d'augmenter les faux positifs si cela réduit les faux négatifs coûteux, car contacter un client faussement à risque coûte moins cher que perdre un churner non détecté.

Les résultats complets sont disponibles dans `reports/imbalance_comparison.csv`.
