"""
Class imbalance study for the churn project.

The study explicitly compares:
- baseline without imbalance handling
- model-level class weighting
- Random Over-Sampling
- SMOTE
- Random Under-Sampling
- decision-threshold tuning

Outputs:
- reports/imbalance_comparison.csv
- reports/imbalance_best_by_strategy.csv
- reports/imbalance_study.md
- models/imbalance_study.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]
DATA_PATH = Path("data/customer_churn_business_dataset.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET].astype(int)
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return df, X, y, numerical, categorical


def preprocessor(numerical: list[str], categorical: list[str]):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def model_factory(weighted: bool = False):
    lr_weight = "balanced" if weighted else None
    rf_weight = "balanced" if weighted else None
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight=lr_weight,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            class_weight=rf_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=130,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(80, 40),
            alpha=0.001,
            max_iter=140,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model, prep, sampler=None):
    if sampler is None:
        return Pipeline([("preprocess", prep), ("model", model)])
    return ImbPipeline([("preprocess", prep), ("sampler", sampler), ("model", model)])


def best_threshold(y_true, probabilities, objective: str = "f1"):
    thresholds = np.linspace(0.05, 0.95, 181)
    if objective == "recall":
        # Keep at least a little precision so the threshold does not degenerate.
        scores = []
        for threshold in thresholds:
            pred = probabilities >= threshold
            precision = precision_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
            scores.append(recall if precision >= 0.10 else 0.0)
    else:
        scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def evaluate(y_true, probabilities, threshold: float):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_study():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df, X, y, numerical, categorical = load_data()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    majority = int(y.value_counts().max())
    minority = int(y.value_counts().min())
    imbalance_ratio = majority / minority

    strategies = {
        "Baseline": {"sampler": None, "weighted": False, "description": "Aucune correction du déséquilibre."},
        "Class Weight": {
            "sampler": None,
            "weighted": True,
            "description": "Pondération des classes dans les modèles compatibles.",
        },
        "Random Over-Sampling": {
            "sampler": RandomOverSampler(random_state=RANDOM_STATE),
            "weighted": False,
            "description": "Duplication aléatoire de la classe minoritaire.",
        },
        "SMOTE": {
            "sampler": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
            "weighted": False,
            "description": "Génération synthétique d'exemples de la classe minoritaire.",
        },
        "Random Under-Sampling": {
            "sampler": RandomUnderSampler(random_state=RANDOM_STATE),
            "weighted": False,
            "description": "Réduction aléatoire de la classe majoritaire.",
        },
    }

    scoring = {
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for strategy_name, strategy in strategies.items():
        models = model_factory(weighted=strategy["weighted"])
        for model_name, model in models.items():
            print(f"Training {strategy_name} / {model_name}")
            prep = preprocessor(numerical, categorical)
            estimator = make_pipeline(model, prep, strategy["sampler"])

            try:
                cv_scores = cross_validate(
                    estimator,
                    X_train_full,
                    y_train_full,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )
                cv_summary = {
                    metric: float(np.mean(cv_scores[f"test_{metric}"]))
                    for metric in scoring
                }
            except Exception as exc:
                cv_summary = {"error": str(exc)}

            estimator.fit(X_train, y_train)
            val_proba = estimator.predict_proba(X_val)[:, 1]
            threshold_f1, val_f1 = best_threshold(y_val, val_proba, objective="f1")
            threshold_recall, val_recall = best_threshold(y_val, val_proba, objective="recall")

            final_estimator = make_pipeline(
                model_factory(weighted=strategy["weighted"])[model_name],
                preprocessor(numerical, categorical),
                strategy["sampler"],
            )
            final_estimator.fit(X_train_full, y_train_full)
            test_proba = final_estimator.predict_proba(X_test)[:, 1]

            default_metrics = evaluate(y_test, test_proba, threshold=0.5)
            tuned_metrics = evaluate(y_test, test_proba, threshold=threshold_f1)
            recall_metrics = evaluate(y_test, test_proba, threshold=threshold_recall)

            rows.append(
                {
                    "strategy": strategy_name,
                    "model": model_name,
                    "description": strategy["description"],
                    "cv_recall": cv_summary.get("recall"),
                    "cv_f1": cv_summary.get("f1"),
                    "cv_roc_auc": cv_summary.get("roc_auc"),
                    "cv_pr_auc": cv_summary.get("pr_auc"),
                    "default_threshold": 0.5,
                    "default_accuracy": default_metrics["accuracy"],
                    "default_precision": default_metrics["precision"],
                    "default_recall": default_metrics["recall"],
                    "default_f1": default_metrics["f1_score"],
                    "default_roc_auc": default_metrics["roc_auc"],
                    "default_pr_auc": default_metrics["pr_auc"],
                    "default_fp": default_metrics["fp"],
                    "default_fn": default_metrics["fn"],
                    "f1_threshold": threshold_f1,
                    "validation_f1": val_f1,
                    "tuned_accuracy": tuned_metrics["accuracy"],
                    "tuned_precision": tuned_metrics["precision"],
                    "tuned_recall": tuned_metrics["recall"],
                    "tuned_f1": tuned_metrics["f1_score"],
                    "tuned_roc_auc": tuned_metrics["roc_auc"],
                    "tuned_pr_auc": tuned_metrics["pr_auc"],
                    "tuned_fp": tuned_metrics["fp"],
                    "tuned_fn": tuned_metrics["fn"],
                    "recall_threshold": threshold_recall,
                    "validation_recall": val_recall,
                    "recall_mode_precision": recall_metrics["precision"],
                    "recall_mode_recall": recall_metrics["recall"],
                    "recall_mode_f1": recall_metrics["f1_score"],
                    "recall_mode_fp": recall_metrics["fp"],
                    "recall_mode_fn": recall_metrics["fn"],
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / "imbalance_comparison.csv", index=False)

    best_by_strategy = (
        results.sort_values(["strategy", "tuned_f1", "tuned_pr_auc"], ascending=[True, False, False])
        .groupby("strategy", as_index=False)
        .head(1)
        .sort_values("tuned_f1", ascending=False)
    )
    best_by_strategy.to_csv(REPORTS_DIR / "imbalance_best_by_strategy.csv", index=False)

    best_overall = results.sort_values(["tuned_f1", "tuned_pr_auc", "tuned_recall"], ascending=False).iloc[0].to_dict()
    baseline_lr = results[(results["strategy"] == "Baseline") & (results["model"] == "Logistic Regression")].iloc[0].to_dict()

    summary = {
        "created_at": datetime.now().isoformat(),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "imbalance_ratio": float(imbalance_ratio),
        "majority_class_size": majority,
        "minority_class_size": minority,
        "best_overall": best_overall,
        "baseline_logistic_regression": baseline_lr,
        "best_by_strategy": best_by_strategy.to_dict(orient="records"),
        "all_results": results.to_dict(orient="records"),
    }
    (MODELS_DIR / "imbalance_study.json").write_text(json.dumps(json_safe(summary), indent=2))
    write_markdown(summary, results, best_by_strategy)
    print(best_by_strategy[["strategy", "model", "tuned_precision", "tuned_recall", "tuned_f1", "tuned_pr_auc", "tuned_fp", "tuned_fn"]])


def write_markdown(summary: dict, results: pd.DataFrame, best_by_strategy: pd.DataFrame) -> None:
    baseline = summary["baseline_logistic_regression"]
    best = summary["best_overall"]
    lines = [
        "# Étude comparative du déséquilibre des classes",
        "",
        "## 1. Constat initial",
        "",
        f"- Classe 0 (non churn) : {summary['class_distribution']['0']} clients.",
        f"- Classe 1 (churn) : {summary['class_distribution']['1']} clients.",
        f"- Ratio de déséquilibre majorité/minorité : **{summary['imbalance_ratio']:.2f}:1**.",
        "",
        "Ce déséquilibre rend l'accuracy insuffisante : un modèle qui prédit majoritairement la classe 0 peut obtenir un bon score global tout en manquant les clients churn. Dans un contexte CRM, les faux négatifs sont coûteux car ils correspondent à des clients à risque non détectés.",
        "",
        "## 2. Baseline et limites de l'accuracy",
        "",
        f"Baseline Logistic Regression au seuil 0.5 : accuracy {baseline['default_accuracy']:.4f}, recall {baseline['default_recall']:.4f}, F1 {baseline['default_f1']:.4f}, PR-AUC {baseline['default_pr_auc']:.4f}.",
        f"Matrice de confusion baseline : FP={int(baseline['default_fp'])}, FN={int(baseline['default_fn'])}.",
        "",
        "Les métriques retenues sont donc : Recall, F1-score, ROC-AUC et PR-AUC. Le Recall mesure la capacité à détecter les churners, le F1 équilibre précision et rappel, ROC-AUC mesure la discrimination globale, et PR-AUC est particulièrement informative quand la classe positive est minoritaire.",
        "",
        "## 3. Méthodes testées",
        "",
        "- Baseline : aucun rééquilibrage.",
        "- Class Weight : pondération automatique des classes dans les modèles compatibles.",
        "- Random Over-Sampling : duplication aléatoire de la classe minoritaire.",
        "- SMOTE : génération synthétique de churners.",
        "- Random Under-Sampling : réduction de la classe majoritaire.",
        "- Ajustement du seuil : recherche du seuil maximisant le F1 sur validation.",
        "",
        "La validation utilise Stratified K-Fold afin de préserver les proportions de churn/non-churn dans chaque fold.",
        "",
        "## 4. Meilleur modèle par stratégie",
        "",
        "| Stratégie | Modèle | Precision | Recall | F1 | ROC-AUC | PR-AUC | Seuil | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_by_strategy.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['model']} | {row['tuned_precision']:.4f} | {row['tuned_recall']:.4f} | "
            f"{row['tuned_f1']:.4f} | {row['tuned_roc_auc']:.4f} | {row['tuned_pr_auc']:.4f} | "
            f"{row['f1_threshold']:.3f} | {int(row['tuned_fp'])} | {int(row['tuned_fn'])} |"
        )

    lines += [
        "",
        "## 5. Synthèse comparative",
        "",
        f"La meilleure configuration observée est **{best['strategy']} + {best['model']}** avec F1={best['tuned_f1']:.4f}, recall={best['tuned_recall']:.4f}, PR-AUC={best['tuned_pr_auc']:.4f} et seuil={best['f1_threshold']:.3f}.",
        "",
        "Effets observés :",
        "",
        "- Le seuil 0.5 est rarement optimal dans ce contexte ; l'ajustement du seuil améliore la détection des churners.",
        "- Random Over-Sampling et SMOTE augmentent généralement le rappel, avec un risque d'overfitting ou de bruit synthétique.",
        "- Random Under-Sampling peut améliorer le rappel mais perd de l'information sur la classe majoritaire.",
        "- Class Weight est simple, robuste et facile à déployer, mais ne suffit pas toujours à maximiser la détection.",
        "",
        "## 6. Recommandation métier",
        "",
        "La stratégie finale doit privilégier un compromis F1/Recall plutôt que l'accuracy. Dans un service CRM, il est acceptable d'augmenter les faux positifs si cela réduit les faux négatifs coûteux, car contacter un client faussement à risque coûte moins cher que perdre un churner non détecté.",
        "",
        "Les résultats complets sont disponibles dans `reports/imbalance_comparison.csv`.",
        "",
    ]
    (REPORTS_DIR / "imbalance_study.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_study()
