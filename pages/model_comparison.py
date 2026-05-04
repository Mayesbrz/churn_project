"""
🤖 Model Comparison Page - Streamlit Dashboard
Compares Random Forest vs MLP Deep Learning predictions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import ClientPredictionRequest
from app.utils import predict_churn, predict_churn_mlp

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 Model Comparison",
    page_icon="⚖️",
    layout="wide"
)

st.title("🤖 ML vs Deep Learning Comparison")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_mlp_metadata():
    with open('models/mlp_metadata.json') as f:
        return json.load(f)

@st.cache_resource
def load_rf_metadata():
    with open('models/model_metadata.json') as f:
        return json.load(f)

mlp_metadata = load_mlp_metadata()
rf_metadata = load_rf_metadata()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Model Comparison Overview
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("📊 Model Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("### 🌲 Random Forest (Production Model)")
    st.info("""
    **Type:** Ensemble of Decision Trees
    - **Framework:** scikit-learn
    - **N Estimators:** 100 trees
    - **Max Depth:** 15
    - **Hyperparameter Optimization:** Manual tuning
    """)
    
    # RF Metrics
    rf_metrics = {
        'Accuracy': 0.8975,
        'Precision': 0.851,
        'Recall': 0.793,
        'F1-Score': 0.8206,
        'ROC-AUC': 0.7914,
        'Specificity': 0.853
    }
    
    st.write("**Performance Metrics:**")
    for metric, value in rf_metrics.items():
        st.metric(metric, f"{value:.4f}")

with col2:
    st.write("### 🧠 MLP Deep Learning")
    st.info("""
    **Type:** Multi-Layer Perceptron
    - **Framework:** TensorFlow/Keras
    - **Hidden Layers:** [128, 64, 32]
    - **Activation:** ReLU (hidden) + Sigmoid (output)
    - **Regularization:** Batch Norm + Dropout
    """)
    
    # MLP Metrics
    mlp_metrics = mlp_metadata['performance_metrics']
    mlp_perf = {
        'Accuracy': mlp_metrics['accuracy'],
        'Precision': mlp_metrics['precision'],
        'Recall': mlp_metrics['recall'],
        'F1-Score': mlp_metrics['f1_score'],
        'ROC-AUC': mlp_metrics['roc_auc'],
        'Specificity': mlp_metrics['specificity']
    }
    
    st.write("**Performance Metrics:**")
    for metric, value in mlp_perf.items():
        st.metric(metric, f"{value:.4f}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Prediction Comparison
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("🔮 Single Client Prediction Comparison")

# Create example client data
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (months)", 0, 120, 24)
    monthly_logins = st.slider("Monthly Logins", 0, 50, 10)

with col2:
    csat = st.slider("CSAT Score", 1.0, 5.0, 3.5)
    nps = st.slider("NPS Score", 0, 100, 45)
    support_tickets = st.slider("Support Tickets", 0, 20, 2)

with col3:
    total_revenue = st.slider("Total Revenue (€)", 0.0, 5000.0, 500.0)
    payment_failures = st.slider("Payment Failures", 0, 10, 1)
    escalations = st.slider("Escalations", 0, 10, 0)

# Create prediction button
if st.button("🔮 Compare Model Predictions"):
    # Build client data
    client_data = ClientPredictionRequest(
        customer_id="COMPARE_001",
        gender="Male",
        country="France",
        city="Paris",
        customer_segment="Standard",
        signup_channel="Web",
        contract_type="Monthly",
        payment_method="Credit Card",
        discount_applied="No",
        price_increase_last_3m="No",
        complaint_type="None",
        survey_response="Positive",
        age=age,
        tenure_months=tenure,
        monthly_logins=monthly_logins,
        weekly_active_days=5,
        avg_session_time=45,
        features_used=15,
        usage_growth_rate=10,
        last_login_days_ago=2,
        monthly_fee=50,
        total_revenue=total_revenue,
        payment_failures=payment_failures,
        support_tickets=support_tickets,
        avg_resolution_time=24,
        csat_score=csat,
        escalations=escalations,
        email_open_rate=0.6,
        marketing_click_rate=0.3,
        nps_score=nps,
        referral_count=2
    )
    
    # Get predictions
    try:
        rf_result = predict_churn(client_data)
        mlp_result = predict_churn_mlp(client_data)
        
        # Display comparison
        col_rf, col_mlp = st.columns(2)
        
        with col_rf:
            st.write("### 🌲 Random Forest Prediction")
            st.metric("Churn Prediction", "🔴 YES" if rf_result['churn_prediction'] else "🟢 NO",
                     delta=f"{rf_result['churn_probability']*100:.1f}% probability")
            st.metric("Risk Level", rf_result['risk_level'])
            st.metric("Confidence", f"{rf_result['confidence']*100:.1f}%")
            
            st.write("**Recommendations:**")
            for i, rec in enumerate(rf_result['recommendations'][:3], 1):
                st.write(f"{i}. {rec}")
        
        with col_mlp:
            st.write("### 🧠 MLP Prediction")
            st.metric("Churn Prediction", "🔴 YES" if mlp_result['churn_prediction'] else "🟢 NO",
                     delta=f"{mlp_result['churn_probability']*100:.1f}% probability")
            st.metric("Risk Level", mlp_result['risk_level'])
            st.metric("Confidence", f"{mlp_result['confidence']*100:.1f}%")
            
            st.write("**Recommendations:**")
            for i, rec in enumerate(mlp_result['recommendations'][:3], 1):
                st.write(f"{i}. {rec}")
        
        # Agreement Analysis
        st.markdown("---")
        st.subheader("📈 Model Agreement Analysis")
        
        agreement = rf_result['churn_prediction'] == mlp_result['churn_prediction']
        prob_diff = abs(rf_result['churn_probability'] - mlp_result['churn_probability'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ AGREE" if agreement else "❌ DISAGREE"
            st.metric("Agreement", status)
        
        with col2:
            st.metric("Probability Difference", f"{prob_diff*100:.1f}%")
        
        with col3:
            consensus = "HIGH" if prob_diff < 0.1 else "MEDIUM" if prob_diff < 0.3 else "LOW"
            st.metric("Consensus Level", consensus)
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Probability Comparison Chart")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Random Forest', 'MLP'],
            y=[rf_result['churn_probability'], mlp_result['churn_probability']],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f"{rf_result['churn_probability']*100:.1f}%", 
                  f"{mlp_result['churn_probability']*100:.1f}%"],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Churn Probability Comparison",
            xaxis_title="Model",
            yaxis_title="Churn Probability",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Architectural Comparison
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("🏗️ Architectural Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("### 🌲 Random Forest Architecture")
    st.write("""
    **Ensemble Method:**
    - 100 Decision Trees trained in parallel
    - Each tree: max_depth=15
    - Feature sampling at each split
    - Bagging (bootstrap aggregating)
    
    **Strengths:**
    - ✅ Handles tabular data well
    - ✅ Fast training
    - ✅ Excellent feature importance
    - ✅ Robust to outliers
    - ✅ Good default parameters
    
    **Weaknesses:**
    - ❌ Larger memory footprint
    - ❌ Harder to update incrementally
    - ❌ Less suitable for very large datasets (100M+ rows)
    """)

with col2:
    st.write("### 🧠 MLP Deep Learning")
    st.write("""
    **Neural Network:**
    - Input: 31 features
    - Hidden 1: 128 neurons + BatchNorm + Dropout(0.3)
    - Hidden 2: 64 neurons + BatchNorm + Dropout(0.3)
    - Hidden 3: 32 neurons + BatchNorm + Dropout(0.2)
    - Output: Sigmoid (binary classification)
    
    **Strengths:**
    - ✅ Can learn complex interactions
    - ✅ Scales to large datasets
    - ✅ Multiple regularization techniques
    - ✅ Transfer learning possible
    
    **Weaknesses:**
    - ❌ Needs more data (100k+)
    - ❌ Black box model
    - ❌ Hyperparameter tuning required
    - ❌ Risk of overfitting
    """)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Recommendation
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("💡 Model Selection Recommendation")

recommendation = st.radio(
    "Select scenario:",
    [
        "Production - Stability & Performance",
        "Research - Understanding DL capabilities",
        "Ensemble - Combined predictions"
    ]
)

if recommendation == "Production - Stability & Performance":
    st.success("""
    ### ✅ Recommendation: **RANDOM FOREST**
    
    **Reasons:**
    1. **Performance:** 89.75% accuracy, 0.7914 ROC-AUC
    2. **Stability:** Robust on small datasets (7,043 samples)
    3. **Interpretability:** Clear feature importance scores
    4. **Deployment:** Easy to serve and monitor
    5. **Maintenance:** Simple to retrain and update
    
    **Action Items:**
    - Use RF as primary production model
    - Monitor predictions with regular retraining (quarterly)
    - Use feature importance for business insights
    - Implement A/B testing for retention actions
    """)

elif recommendation == "Research - Understanding DL capabilities":
    st.info("""
    ### 📊 Recommendation: **MLP FOR RESEARCH**
    
    **Value:**
    1. **Validation:** Demonstrates Deep Learning knowledge
    2. **Learning:** Understand limitations on small datasets
    3. **Future:** Pathway for larger dataset scenarios
    4. **Comparison:** Scientific analysis of ML vs DL
    
    **Use Cases:**
    - Academic research and publication
    - Knowledge building and skill development
    - Experimentation with architectural improvements
    - Dataset collection for future DL-ready scenarios
    """)

else:
    st.warning("""
    ### 🔄 Recommendation: **ENSEMBLE APPROACH**
    
    **Combined Strategy:**
    1. Use RF for primary decision (89.75% accuracy)
    2. Use MLP confidence as secondary validation
    3. Alert when models disagree (consensus analysis)
    4. Weight predictions: 70% RF + 30% MLP
    
    **Benefits:**
    - Better robustness through ensemble
    - Catch edge cases both models miss
    - Monitor model drift separately
    - Gradual migration path to DL-only (when data grows)
    """)

st.markdown("---")

# Footer
st.markdown("""
---
**Note:** This comparison demonstrates the importance of choosing the right model
for the right problem. Random Forest excels on structured, tabular data with moderate
sample sizes. Deep Learning shines with large datasets and unstructured data.
Both approaches are valuable in the modern ML toolkit!
""")
