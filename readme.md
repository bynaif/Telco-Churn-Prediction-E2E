# 📡 Telco Churn Prediction — End to End ML Product

## ❗ Problem Statement

Customer churn is the **silent killer of telecom revenue**. Industry research shows:

- Acquiring a new customer costs **5–7x more** than retaining an existing one
- A **5% reduction in churn** can increase profits by **25–95%** (Harvard Business Review)
- The average telecom company loses **15–25% of its customer base annually**
- Churned customers are rarely won back — the window to act is **before** they leave

This product gives telecom retention teams a real-time churn scoring engine — predicting which customers are about to leave, explaining exactly why, and recommending targeted actions to keep them.

---

## 🌍 Live Demo

👉 **[https://telco-churn-prediction-e2e.onrender.com](https://telco-churn-prediction-e2e.onrender.com)**

---

## 💡 Use Case & ROI

| Scenario | Without Tool | With Tool |
|---|---|---|
| Churn detection | Reactive (after customer leaves) | Proactive (before they leave) |
| Screening time | Days (analyst team needed) | < 2 seconds per customer |
| Action targeting | Blanket discounts to all | Precision interventions on at-risk only |
| Explainability | Gut feeling | SHAP-driven feature impact |
| Detection threshold | Standard 0.5 | Tuned to 0.3 (recall-optimized) |

**Target users:**
- Telecom retention and CRM teams
- Customer success managers
- Revenue operations analysts
- Data teams building churn dashboards

---

## 📁 Project Structure

```
Telco-Churn-Prediction-E2E/
├── data/
│   └── telco_churn.csv              ← Kaggle Telco Customer Churn dataset
├── model/
│   └── model.pkl                    ← Trained LightGBM pipeline
├── notebook/
│   └── notebook.ipynb               ← EDA, training, evaluation
├── src/
│   ├── __init__.py
│   ├── backend/
│   │   ├── __init__.py
│   │   └── main.py                  ← FastAPI app
│   └── frontend/
│       ├── __init__.py
│       └── app.py                   ← Streamlit dashboard
├── .github/
│   └── workflows/
│       └── deploy.yml               ← CI/CD pipeline
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## 🧠 ML Pipeline

- **Dataset:** Kaggle Telco Customer Churn (7,043 customers, 20 features)
- **Models Compared:** Logistic Regression, LightGBM, XGBoost, CatBoost, Random Forest
- **Best Model:** LightGBM — selected for best Recall/F1 balance with lowest false alarms
- **Threshold Tuning:** Custom threshold of **0.3** (vs default 0.5) — optimized for Recall over Precision
- **Explainability:** SHAP TreeExplainer — feature impact per prediction
- **Serialization:** joblib

> **Why Recall over Precision?** In churn prevention, a false negative (missing a customer who churns) costs you the entire customer lifetime value. A false positive (flagging someone who stays) costs only a small retention offer. Tuning from 0.5 → 0.3 catches significantly more churners even at the cost of more false alarms — a justified business trade-off.

### Model Comparison (Default Threshold)

| Model | Recall | Precision | F1 Score |
|---|---|---|---|
| **Logistic Regression** | **0.801** | 0.513 | 0.626 |
| LightGBM | 0.738 | 0.528 | 0.616 |
| XGBoost | 0.682 | 0.540 | 0.603 |
| CatBoost | 0.520 | 0.650 | 0.577 |
| Random Forest | 0.482 | 0.630 | 0.546 |

### Final Model Performance (Threshold = 0.3)

| Model | Threshold | Precision | Recall | F1 Score | Missed Churners (FN) |
|---|---|---|---|---|---|
| Logistic Regression | 0.3 | 0.429 | 0.928 | 0.587 | 27 |
| **LightGBM** | **0.3** | **0.459** | **0.861** | **0.599** | **52** |

> LightGBM was selected over Logistic Regression despite fewer missed churners, because it delivers **better precision at comparable recall** — meaning fewer wasted retention offers while still catching the majority of at-risk customers.

**Input Features:**

| Feature | Description |
|---|---|
| gender | Customer gender |
| SeniorCitizen | Senior citizen flag (0/1) |
| Partner | Has a partner (Yes/No) |
| Dependents | Has dependents (Yes/No) |
| tenure | Months with the company |
| PhoneService | Has phone service |
| MultipleLines | Multiple lines subscribed |
| InternetService | DSL / Fiber optic / None |
| OnlineSecurity | Online security add-on |
| OnlineBackup | Online backup add-on |
| DeviceProtection | Device protection add-on |
| TechSupport | Tech support add-on |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling | Paperless billing enabled |
| PaymentMethod | Payment method type |
| MonthlyCharges | Monthly bill amount ($) |
| TotalCharges | Total spend to date ($) |

---

## 🎨 Dashboard Features

- **Churn Risk Banner** — High / Low risk classification with probability
- **Risk Gauge Meter** — Visual probability indicator (0–100%)
- **SHAP Explainability** — Bar chart showing top drivers of the prediction
- **Retention Action Engine** — Rule-based recommendations mapped to top SHAP features
- **ROI-Tagged Actions** — Each recommendation tagged Very High / High / Medium / Low ROI
- **Sidebar Customer Profiler** — Full 20-feature input panel

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | LightGBM, scikit-learn |
| Explainability | SHAP TreeExplainer |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| Serialization | Joblib |
| Containerization | Docker |
| Deployment | Render |
| CI/CD | GitHub Actions |

---

## 💪 Strengths

- **Recall-optimized** — catches 86% of churners before they leave, business-justified decision
- **Explainable AI** — SHAP shows exactly which features drove each prediction
- **Retention action engine** — not just a score, but what to do about it
- **ROI-tagged recommendations** — prioritised by business impact, not just model output
- **Full MLOps pipeline** — Docker + CI/CD + live deployment

---

## ⚠️ Limitations & Honest Notes

- **Static dataset** — 7,043 samples from a single snapshot. No temporal drift detection yet
- **No model monitoring** — no automated alerts if performance degrades over time. Next step: Evidently AI integration
- **No authentication** — API is open. Production would require API keys or OAuth
- **Rule-based recommendations** — retention actions are heuristic, not learned from outcome data
- **CSV-free logging** — predictions are not persisted. Production would use PostgreSQL with timestamps and user tracking

---

## 🚀 How to Run

### Option A — Live Demo
👉 **[https://telco-churn-prediction-e2e.onrender.com](https://telco-churn-prediction-e2e.onrender.com)**

### Option B — Docker
```bash
docker build -t telco-churn .
docker run -p 8000:8000 -p 8001:8001 telco-churn
```
Open `http://localhost:8000`

### Option C — Local
```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 — FastAPI
uvicorn src.backend.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 — Streamlit
streamlit run src/frontend/app.py --server.port 8000
```

---

## 📊 Sample Prediction

**Input:**
```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 0,
  "Dependents": 0,
  "tenure": 2,
  "PhoneService": 1,
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": 1,
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 171.0
}
```

**Output:**
```json
{
  "churn": true,
  "churn_probability": 0.847,
  "shap_values": [
    {"feature": "Contract", "impact": 0.312},
    {"feature": "tenure", "impact": 0.287},
    {"feature": "MonthlyCharges", "impact": 0.201}
  ]
}
```

---

## 👤 Author

**Mohammad Naif** — Data Science Undergrad | ML Engineer in Progress
