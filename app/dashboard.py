"""
Decision dashboard for customer churn prediction.

The dashboard loads the local trained model directly. The API remains optional
and is not required for the Streamlit experience.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_business_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.json"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
SHAP_GLOBAL_PATH = BASE_DIR / "reports" / "shap_global_importance.csv"
SHAP_LOCAL_PATH = BASE_DIR / "reports" / "shap_local_examples.csv"
SHAP_BAR_PATH = BASE_DIR / "reports" / "shap_summary_bar.png"
SHAP_BEESWARM_PATH = BASE_DIR / "reports" / "shap_beeswarm.png"

st.set_page_config(
    page_title="Retention Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #152235;
            --muted: #64748b;
            --line: #e2e8f0;
            --panel: #ffffff;
            --soft: #f6f8fb;
            --blue: #2563eb;
            --teal: #0f766e;
            --red: #dc2626;
        }
        .stApp {
            background: #f6f8fb;
        }
        section[data-testid="stSidebar"] {
            background: #0f1f33;
        }
        section[data-testid="stSidebar"] * {
            color: #e5edf8 !important;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            background: linear-gradient(135deg, #12233b 0%, #17426f 55%, #0f766e 100%);
            border-radius: 8px;
            padding: 26px 30px;
            color: white;
            margin-bottom: 22px;
        }
        .hero h1 {
            font-size: 34px;
            line-height: 1.08;
            margin: 0 0 8px 0;
            color: white;
        }
        .hero p {
            color: #dbeafe;
            max-width: 820px;
            margin: 0;
            font-size: 15px;
        }
        .section-title {
            color: var(--ink);
            font-size: 20px;
            font-weight: 750;
            margin: 10px 0 8px 0;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b;
        }
        div[data-testid="stMetricValue"] {
            color: #152235;
            font-size: 25px;
        }
        .insight-box {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px 18px;
            margin-top: 8px;
        }
        .risk-low {
            color: #047857;
            font-weight: 800;
        }
        .risk-medium {
            color: #b45309;
            font-weight: 800;
        }
        .risk-high {
            color: #b91c1c;
            font-weight: 800;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            height: 42px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    comparison = json.loads(COMPARISON_PATH.read_text())
    features = json.loads(FEATURES_PATH.read_text())
    return model, metadata, comparison, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_shap_outputs():
    global_df = pd.read_csv(SHAP_GLOBAL_PATH) if SHAP_GLOBAL_PATH.exists() else pd.DataFrame()
    local_df = pd.read_csv(SHAP_LOCAL_PATH) if SHAP_LOCAL_PATH.exists() else pd.DataFrame()
    return global_df, local_df


def pct(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"{value:,.0f} €".replace(",", " ")


def mode_or_first(series: pd.Series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else series.dropna().iloc[0]


def baseline_client(df: pd.DataFrame, feature_order: list[str]) -> dict:
    row = {}
    for col in feature_order:
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median())
        else:
            row[col] = mode_or_first(df[col])
    return row


def risk_label(probability: float, threshold: float) -> tuple[str, str]:
    if probability >= max(0.7, threshold):
        return "Risque élevé", "risk-high"
    if probability >= threshold:
        return "Risque moyen", "risk-medium"
    return "Risque faible", "risk-low"


def prediction_form(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    st.markdown('<div class="section-title">Simulation client</div>', unsafe_allow_html=True)
    st.caption("Les autres variables sont complétées automatiquement avec un profil client médian du dataset.")

    client = baseline_client(df, feature_order)

    col1, col2 = st.columns(2)
    with col1:
        client["customer_segment"] = st.selectbox("Segment client", sorted(df["customer_segment"].dropna().unique()))
        client["contract_type"] = st.selectbox("Type de contrat", sorted(df["contract_type"].dropna().unique()))
        client["tenure_months"] = st.slider("Ancienneté (mois)", 0, 72, 18)
        client["monthly_fee"] = st.slider("Abonnement mensuel (€)", 0.0, 300.0, 50.0, 5.0)
        client["total_revenue"] = max(client["monthly_fee"] * max(client["tenure_months"], 1), client["monthly_fee"])

    with col2:
        client["monthly_logins"] = st.slider("Connexions mensuelles", 0, 40, 10)
        client["csat_score"] = st.slider("Satisfaction CSAT", 1.0, 5.0, 3.5, 0.1)
        client["payment_failures"] = st.slider("Échecs de paiement", 0, 10, 1)
        client["support_tickets"] = st.slider("Tickets support", 0, 25, 2)
        client["nps_score"] = st.slider("NPS", -100, 100, 20)

    client["weekly_active_days"] = min(7, max(0, round(client["monthly_logins"] / 4)))
    client["last_login_days_ago"] = max(0, int(18 - client["monthly_logins"] / 2))
    client["avg_session_time"] = max(5.0, float(client.get("avg_session_time", 30)))
    client["features_used"] = max(1, int(client.get("features_used", 6)))
    client["usage_growth_rate"] = -0.2 if client["monthly_logins"] < 5 else 0.05
    client["email_open_rate"] = max(0.0, min(1.0, float(client.get("email_open_rate", 0.55))))
    client["marketing_click_rate"] = max(0.0, min(1.0, float(client.get("marketing_click_rate", 0.2))))
    client["avg_resolution_time"] = float(client.get("avg_resolution_time", 24))
    client["escalations"] = 1 if client["support_tickets"] >= 5 else 0

    return pd.DataFrame([client], columns=feature_order)


def recommendations(client: pd.Series, probability: float, threshold: float) -> list[str]:
    actions = []
    if probability >= max(0.7, threshold):
        actions.append("Contacter le client rapidement avec une offre de rétention ciblée.")
    elif probability >= threshold:
        actions.append("Planifier une action CRM proactive dans la semaine.")
    else:
        actions.append("Maintenir le suivi relationnel standard.")

    if client["csat_score"] < 3:
        actions.append("Traiter le sujet satisfaction : diagnostic support ou appel qualité.")
    if client["monthly_logins"] < 5:
        actions.append("Relancer l'engagement produit avec onboarding ou campagne de réactivation.")
    if client["payment_failures"] >= 2:
        actions.append("Proposer une assistance facturation ou une facilité de paiement.")
    if client["contract_type"] == "Monthly":
        actions.append("Tester une incitation vers un contrat plus long.")
    return actions


def plot_card(fig):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=35),
        font=dict(color="#152235"),
        title_font=dict(size=16, color="#152235"),
    )
    return fig


model, metadata, comparison, features = load_artifacts()
df = load_data()
shap_global, shap_local = load_shap_outputs()

st.markdown(
    f"""
    <div class="hero">
        <h1>Retention Cockpit</h1>
        <p>Suivi du churn, priorisation des clients à risque et simulation en temps réel avec le modèle final : <b>{metadata["model_type"]}</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Pilotage", "Prédiction", "Modèles", "Explicabilité", "Données"], label_visibility="collapsed")
    st.divider()
    st.markdown("### Modèle")
    st.metric("Type", metadata["model_type"])
    st.metric("ROC-AUC", pct(metadata["roc_auc"]))
    st.metric("F1", f"{metadata['f1_score']:.3f}")
    st.metric("Seuil", f"{metadata['threshold']:.3f}")


if page == "Pilotage":
    churn_count = int(df["churn"].sum())
    churn_rate = float(df["churn"].mean())
    revenue_at_risk = float(df.loc[df["churn"] == 1, "monthly_fee"].sum())
    avg_fee_churn = float(df.loc[df["churn"] == 1, "monthly_fee"].mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients", f"{len(df):,}".replace(",", " "))
    col2.metric("Clients churn", f"{churn_count:,}".replace(",", " "), pct(churn_rate))
    col3.metric("MRR à risque observé", money(revenue_at_risk))
    col4.metric("Abonnement moyen churn", money(avg_fee_churn))

    st.markdown('<div class="section-title">Lecture métier</div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])
    with left:
        seg = df.groupby("customer_segment", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
        fig = px.bar(seg, x="customer_segment", y="churn", title="Taux de churn par segment", color="churn", color_continuous_scale=["#99f6e4", "#2563eb"])
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(plot_card(fig), use_container_width=True)
    with right:
        contract = df.groupby("contract_type", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
        fig = px.bar(contract, x="contract_type", y="churn", title="Churn par type de contrat", color="churn", color_continuous_scale=["#bfdbfe", "#dc2626"])
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(plot_card(fig), use_container_width=True)

    col_a, col_b = st.columns([1.15, 0.85])
    with col_a:
        if not shap_global.empty:
            importance = shap_global.rename(columns={"mean_abs_shap": "importance"}).head(10)
            title = "Variables les plus influentes selon SHAP"
        else:
            importance = pd.DataFrame(metadata["feature_importance"]).head(10)
            title = "Variables les plus influentes"
        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title=title,
            color_discrete_sequence=["#0f766e"],
        )
        st.plotly_chart(plot_card(fig), use_container_width=True)
    with col_b:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Priorités opérationnelles**")
        st.write("- Surveiller satisfaction, paiement et engagement.")
        st.write("- Adapter les campagnes selon segment et contrat.")
        st.write("- Utiliser le seuil métier plutôt que l'accuracy seule.")
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Prédiction":
    left, right = st.columns([0.95, 1.05])
    with left:
        client_df = prediction_form(df, features["all_features"])
        predict = st.button("Prédire le risque de churn", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Résultat de prédiction</div>', unsafe_allow_html=True)
        if predict:
            probability = float(model.predict_proba(client_df[features["all_features"]])[:, 1][0])
            prediction = int(probability >= metadata["threshold"])
            expected_loss = float(client_df.loc[0, "monthly_fee"] * probability)
            label, css_class = risk_label(probability, metadata["threshold"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilité churn", pct(probability))
            c2.metric("Décision", "À risque" if prediction else "Non prioritaire")
            c3.metric("MRR à risque", f"{expected_loss:.2f} €")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 34}},
                    title={"text": "Score de churn"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563eb"},
                        "steps": [
                            {"range": [0, metadata["threshold"] * 100], "color": "#dcfce7"},
                            {"range": [metadata["threshold"] * 100, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#111827", "width": 4},
                            "thickness": 0.8,
                            "value": metadata["threshold"] * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=310, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<div class="insight-box"><span class="{css_class}">{label}</span><br><br>', unsafe_allow_html=True)
            for action in recommendations(client_df.iloc[0], probability, metadata["threshold"]):
                st.write(f"- {action}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="insight-box">
                    <b>Prêt pour une simulation.</b><br><br>
                    Ajuste quelques paramètres client, puis lance la prédiction.
                    Le modèle complète automatiquement les autres variables avec un profil médian.
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "Modèles":
    rows = []
    for item in comparison["models"]:
        rows.append(
            {
                "Modèle": item["model"],
                "Accuracy": item["accuracy"],
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1": item["f1_score"],
                "ROC-AUC": item["roc_auc"],
                "Seuil": item["threshold"],
            }
        )
    results_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    st.markdown('<div class="section-title">Comparaison quantitative</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    chart_df = results_df.melt(id_vars="Modèle", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    fig = px.bar(
        chart_df,
        x="Modèle",
        y="value",
        color="variable",
        barmode="group",
        title="Performance des modèles",
        color_discrete_sequence=["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#64748b"],
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(plot_card(fig), use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-box">
            <b>Choix du modèle.</b> Le modèle final est <b>{metadata["model_type"]}</b>,
            sélectionné sur le ROC-AUC. Le seuil de décision est ajusté sur validation
            pour éviter un F1 nul dans un contexte de classes déséquilibrées.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Explicabilité":
    st.markdown('<div class="section-title">Explicabilité du modèle avec SHAP</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-box">
            SHAP permet d'expliquer l'influence des variables sur les prédictions du modèle.
            Les valeurs affichées concernent la classe churn. Une valeur SHAP positive pousse
            la prédiction vers un risque plus élevé, tandis qu'une valeur négative réduit ce risque.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if shap_global.empty:
        st.warning("Les fichiers SHAP ne sont pas encore générés. Lancez python shap_analysis.py.")
    else:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            fig = px.bar(
                shap_global.head(15).sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Importance globale SHAP",
                color_discrete_sequence=["#1f4e79"],
            )
            st.plotly_chart(plot_card(fig), use_container_width=True)
        with c2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Top variables globales**")
            for _, row in shap_global.head(6).iterrows():
                st.write(f"- {row['feature']} : {row['mean_abs_shap']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        if SHAP_BEESWARM_PATH.exists():
            st.image(str(SHAP_BEESWARM_PATH), caption="Distribution des effets SHAP sur l'échantillon de test")

        st.markdown('<div class="section-title">Exemples locaux</div>', unsafe_allow_html=True)
        if not shap_local.empty:
            selected = st.selectbox(
                "Client expliqué",
                sorted(shap_local["rank"].unique()),
                format_func=lambda x: f"Exemple {x}",
            )
            local_view = shap_local[shap_local["rank"] == selected].copy()
            probability = float(local_view["predicted_probability"].iloc[0])
            st.metric("Probabilité prédite", pct(probability))
            local_view["impact"] = local_view["shap_value"].map(lambda v: "Augmente le risque" if v > 0 else "Réduit le risque")
            st.dataframe(
                local_view[["feature", "shap_value", "impact"]].rename(
                    columns={"feature": "Variable", "shap_value": "Valeur SHAP", "impact": "Impact"}
                ),
                use_container_width=True,
                hide_index=True,
            )


else:
    st.markdown('<div class="section-title">Exploration du dataset</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", f"{len(df):,}".replace(",", " "))
    col2.metric("Colonnes", len(df.columns))
    col3.metric("Taux churn", pct(float(df["churn"].mean())))

    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

st.caption(f"Dernière mise à jour : {datetime.now():%Y-%m-%d %H:%M}")
