import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

API_URL  = os.getenv("API_URL", "https://localhost:8001")

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# Recommendation engine (rule-based)

RECOMMENDATIONS = {
    "Contract": {
        "action": "Offer incentive to upgrade to a 1 or 2-year contract",
        "roi": "Very High",
        "why": "Month-to-month customers churn 3x more than annual subscribers"
    },
    "tenure": {
        "action": "Trigger onboarding engagement campaign for new customers",
        "roi": "High",
        "why": "Churn risk is highest in the first 6 months"
    },
    "MonthlyCharges": {
        "action": "Offer a personalised discount or a lower-tier plan",
        "roi": "High",
        "why": "Price sensitivity is a top driver in early churn"
    },
    "OnlineSecurity": {
        "action": "Offer free trial of Online Security add-on",
        "roi": "Medium",
        "why": "Customers without security services churn more frequently"
    },
    "TechSupport": {
        "action": "Proactively reach out with tech support assistance",
        "roi": "Medium",
        "why": "Poor support experience is a leading churn signal"
    },
    "InternetService": {
        "action": "Review service quality or offer a speed upgrade",
        "roi": "Medium",
        "why": "Fiber optic customers have high expectations and churn faster"
    },
    "PaymentMethod": {
        "action": "Encourage switch to auto-pay (bank transfer or credit card)",
        "roi": "Medium",
        "why": "Electronic check users have higher churn than auto-pay users"
    },
    "PaperlessBilling": {
        "action": "Confirm customer is comfortable with paperless billing",
        "roi": "Low",
        "why": "Billing confusion can lead to dissatisfaction"
    },
}
 
ROI_COLOR = {
    "Very High": "#ef4444",
    "High":      "#f97316",
    "Medium":    "#eab308",
    "Low":       "#22c55e",
}
 
 
def get_recommendations(shap_values, churn):
    if not churn:
        return []
    recs = []
    for item in shap_values:
        feature = item["feature"]
        impact  = item["impact"]
        if impact > 0 and feature in RECOMMENDATIONS:
            rec = RECOMMENDATIONS[feature].copy()
            rec["feature"] = feature
            rec["impact"]  = impact
            recs.append(rec)
        if len(recs) == 3:
            break
    return recs


# Customer Input
st.sidebar.title("Customer Profile") 
with st.sidebar:
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["No", "Yes"])
    dependents      = st.selectbox("Dependents", ["No", "Yes"])
    phone_service   = st.selectbox("Phone Service", ["No", "Yes"])
    paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])
 
    st.markdown("---")
    tenure          = st.slider("Tenure (months)", 0, 72, 2)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=1.0, value=85.5)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=171.0)
 
    st.markdown("---")
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
 
    st.markdown("---")
    contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment         = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
 
    predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)
 

# Payload Building
def build_payload():
    return {
        "gender":          1 if gender == "Male" else 0,
        "SeniorCitizen":   1 if senior == "Yes" else 0,
        "Partner":         1 if partner == "Yes" else 0,
        "Dependents":      1 if dependents == "Yes" else 0,
        "PhoneService":    1 if phone_service == "Yes" else 0,
        "PaperlessBilling":1 if paperless == "Yes" else 0,
        "tenure":          tenure,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
        "MultipleLines":   multiple_lines,
        "InternetService": internet,
        "OnlineSecurity":  online_security,
        "OnlineBackup":    online_backup,
        "DeviceProtection":device_prot,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract":        contract,
        "PaymentMethod":   payment,
    }

# Main Panel 
st.title("📡 Telco Churn Prediction")
st.caption("Predict customer churn risk and get actionable retention recommendations")
 
if not predict_btn:
    st.info("Configure the customer profile in the sidebar and click **Run Prediction**.")
    st.stop()
 
with st.spinner("Running prediction..."):
    try:
        response = requests.post(f"{API_URL}/predict", json=build_payload(), timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Make sure FastAPI is running.")
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()
 
churn       = data["churn"]
probability = data["churn_probability"]
shap_values = data["shap_values"]


# Result Banner
col1, col2 = st.columns([1, 2])
 
with col1:
    if churn:
        st.error(f"### ⚠️ High Churn Risk\n**{probability * 100:.1f}%** probability")
    else:
        st.success(f"### ✅ Low Churn Risk\n**{probability * 100:.1f}%** probability")
 
with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#ef4444" if churn else "#22c55e"},
            "steps": [
                {"range": [0, 30],   "color": "#dcfce7"},
                {"range": [30, 70],  "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 30
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(t=20, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
 
st.markdown("---")

# Shap chart + recommendation 
col_shap, col_rec = st.columns([3, 2])
 
with col_shap:
    st.subheader("Feature Impact (SHAP)")
 
    df_shap = pd.DataFrame(shap_values).head(10)
    colors  = ["#ef4444" if v > 0 else "#22c55e" for v in df_shap["impact"]]
 
    fig2 = go.Figure(go.Bar(
        x=df_shap["impact"],
        y=df_shap["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df_shap["impact"]],
        textposition="outside"
    ))
    fig2.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="SHAP Impact",
        yaxis={"autorange": "reversed"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔴 Red = increases churn risk &nbsp;&nbsp; 🟢 Green = reduces churn risk")
 
with col_rec:
    st.subheader("Retention Actions")
 
    if not churn:
        st.success("No immediate action needed. Customer is stable.")
    else:
        recs = get_recommendations(shap_values, churn)
        if not recs:
            st.warning("No specific recommendations available.")
        for rec in recs:
            roi_color = ROI_COLOR.get(rec["roi"], "#888")
            st.markdown(f"""
<div style="border-left: 4px solid {roi_color}; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; background: rgba(0,0,0,0.03)">
    <div style="font-size:13px; color: {roi_color}; font-weight:600; margin-bottom:4px">
        {rec['feature']} &nbsp;·&nbsp; ROI: {rec['roi']}
    </div>
    <div style="font-size:15px; font-weight:500; margin-bottom:4px">{rec['action']}</div>
    <div style="font-size:12px; color: #888">{rec['why']}</div>
</div>
""", unsafe_allow_html=True)