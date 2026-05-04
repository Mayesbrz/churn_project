# Système intelligent multi-modèles pour la rétention client

Projet Data Science EFREI M2 - prédiction du churn client, comparaison ML/DL, dashboard Streamlit et API optionnelle.

## État du projet

- Dataset : `data/customer_churn_business_dataset.csv` (10000 clients).
- Cible : `churn` (classification binaire).
- Modèles comparés : Logistic Regression, Random Forest, Gradient Boosting, MLP.
- Modèle final : **Gradient Boosting**.
- Performance test : ROC-AUC **0.8058**, F1 **0.3802**, Recall **0.8480**.
- Le seuil de décision est optimisé sur validation pour éviter le problème `F1 = 0` sur la classe churn.
- Dashboard : autonome, il charge le modèle localement et ne dépend pas de l'API.
- API : disponible en bonus, mais optionnelle.

## Structure

```text
data/                         Dataset
src/                          Modules data/model/evaluation/explainability
models/                       Pipelines et métriques exportés
app/dashboard.py              Dashboard Streamlit
app/api.py                    API FastAPI optionnelle
reports/reports.docx          Rapport corrigé
reports/presentation_churn_retention.pptx  Support de présentation
train_all_models.py           Entraînement reproductible multi-modèles
build_final_report.py         Génération du rapport Word
build_presentation.mjs        Génération du support PPTX
```

## Reproduire l'entraînement

```bash
source .venv/bin/activate
python train_all_models.py
```

Le script effectue : split stratifié, preprocessing dans `Pipeline`, cross-validation, optimisation du seuil, comparaison des modèles et export des artefacts.

## Lancer le dashboard

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

## Lancer l'API optionnelle

```bash
source .venv/bin/activate
uvicorn app.api:app --reload
```

Endpoints principaux : `/health`, `/predict`, `/batch_predict`, `/model/info`.

## Résultats modèles

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | Seuil |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7680 | 0.2124 | 0.4706 | 0.2927 | 0.7245 | 0.600 |
| Random Forest | 0.7320 | 0.2500 | 0.8137 | 0.3825 | 0.7890 | 0.325 |
| Gradient Boosting | 0.7180 | 0.2450 | 0.8480 | 0.3802 | 0.8058 | 0.115 |
| MLP | 0.6940 | 0.1928 | 0.6275 | 0.2949 | 0.7055 | 0.115 |

## Notes méthodologiques

- `customer_id` est exclu des features pour éviter d'apprendre un identifiant non généralisable.
- Les transformations sont apprises uniquement sur le train set via `ColumnTransformer`.
- L'interprétabilité globale utilise la permutation importance du modèle final.
- SHAP reste une amélioration possible pour l'explication locale de chaque prédiction.

## Étude du déséquilibre des classes

Consignes supplémentaires appliquées dans `imbalance_study.py`.

- Classe 0 : 8979 clients.
- Classe 1 : 1021 clients.
- Ratio majorité/minorité : **8.79:1**.
- Métriques utilisées : Recall, F1-score, ROC-AUC et **PR-AUC**.
- Validation : Stratified K-Fold pour préserver les proportions de classes.
- Méthodes comparées : baseline, `class_weight`, Random Over-Sampling, SMOTE, Random Under-Sampling, seuil optimisé.
- Modèle final appliqué au dashboard : **Random Over-Sampling + Random Forest**.
- Seuil final : **0.390**.
- Résultats test : Precision **0.2629**, Recall **0.7990**, F1 **0.3956**, ROC-AUC **0.8010**, PR-AUC **0.2679**.
- Matrice de confusion : FP=457, FN=41, TP=163, TN=1339.

Documentation complète : `reports/imbalance_study.md`.
Résultats complets : `reports/imbalance_comparison.csv`.
Synthèse par méthode : `reports/imbalance_best_by_strategy.csv`.

### Commandes

```bash
source .venv/bin/activate
python imbalance_study.py
python apply_imbalance_final_model.py
```

## SHAP explainability

SHAP a été ajouté pour expliquer le modèle final localement et globalement.

Fichiers générés :

- `reports/shap_global_importance.csv` : importance globale des variables pour la classe churn.
- `reports/shap_local_examples.csv` : exemples de prédictions client expliquées variable par variable.
- `reports/shap_summary_bar.png` : graphique d'importance globale.
- `reports/shap_beeswarm.png` : distribution des effets SHAP.
- `models/shap_explanations.json` : synthèse structurée des explications.

Commande :

```bash
source .venv/bin/activate
python shap_analysis.py
```

Le dashboard contient maintenant une page `Explicabilité` dédiée à SHAP.
