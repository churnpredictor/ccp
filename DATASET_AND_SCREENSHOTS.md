# 📊 Dataset & Application Gallery

---

## 🗂️ Dataset Source

> **Kaggle Dataset:** [Customer Churn Dataset — Muhammad Shahid Azeem](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

The dataset used in this project is sourced from Kaggle and contains **440,833 customer records** with features covering demographics, service usage, billing behaviour, and subscription details.

| Property | Details |
|---|---|
| **Source** | Kaggle |
| **Author** | Muhammad Shahid Azeem |
| **Records** | ~440,833 rows |
| **Features** | Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction |
| **Target** | `Churn` (1 = Churned, 0 = Stayed) |
| **Link** | https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset |

---

## 🌐 Live Application

The dashboard runs locally at: **http://localhost:5000/**

Start the server with:
```bash
python start.py
```

---

## 🖼️ Application Screenshots

### 1. 📊 Dashboard — Overview & KPIs

The main dashboard displays live KPIs: total customers, active vs. churned counts, overall churn rate, and a model performance comparison table.

- **Total Customers:** 440,832
- **Churned:** 249,999
- **Retained:** 190,833
- **Churn Rate:** 56.71%
- **Best Model:** XGBoost (F1 Score: 99.96%)

![Dashboard Overview](static/screenshots/dashboard_top_1779189552146.png)
![Dashboard — Charts & Model Table](static/screenshots/dashboard_bottom_1779189561978.png)

---

### 2. 🔍 Customer History

Search and browse individual customer records by ID, name, or churn date range. Displays a detailed data table with full customer profiles.

![Customer History — Search](static/screenshots/customer_history_1779189583383.png)
![Customer History — Results Table](static/screenshots/customer_history_bottom_1779189593698.png)

---

### 3. 🔮 Single Prediction

Enter any customer's profile details to get an instant churn prediction from the trained ML models.

**Input fields:** Age, Gender, Tenure, Monthly Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction.

![Single Prediction — Form (Top)](static/screenshots/single_prediction_1779189626145.png)
![Single Prediction — Form (Bottom)](static/screenshots/single_prediction_bottom_1779189662737.png)

---

### 4. 📂 Bulk Prediction

Upload a CSV file to run batch predictions across hundreds or thousands of customers simultaneously.

Expected CSV columns match the dataset schema.

![Bulk Prediction — Upload Interface](static/screenshots/bulk_prediction_1779189737337.png)

---

### 5. 📈 Model Analytics

In-depth model performance visualisation including Confusion Matrices, ROC Curves, and comparative Performance metrics for all four models: Decision Tree, Logistic Regression, Random Forest, and XGBoost.

![Model Analytics — Confusion Matrices](static/screenshots/model_analytics_cm_1779189817633.png)
![Model Analytics — ROC / Performance](static/screenshots/model_analytics_cm_bot_1779189854509.png)

---

### 6. 🌳 Decision Tree Playground

Interactive step-through of the Decision Tree logic. Adjust sliders and dropdowns to trace the exact path through the tree for any custom scenario.

![Decision Tree Playground](static/screenshots/decision_tree_1779189932766.png)

---

### 7. ⚖️ Model Comparison

Side-by-side Radar Chart comparison of all four models across Accuracy, Precision, Recall, F1-Score, and AUC. Includes expert summary cards explaining the optimal use case for each model.

![Model Comparison — Radar Chart](static/screenshots/model_comparison_1779190009230.png)
![Model Comparison — Model Cards](static/screenshots/model_comparison_bot_1779190044565.png)

---

### 8. 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis for the Random Forest model. Shows which features most strongly drive churn predictions.

**Top churn drivers identified:**
1. **Support Calls** — highest impact
2. **Total Spend**
3. **Payment Delay**

![SHAP Feature Importance (Top)](static/screenshots/explainability_1779190144239.png)
![SHAP Feature Importance (Key Insights)](static/screenshots/explainability_bot_1779190180653.png)

---

### 9. 📅 Monthly Trends

Select a date range to see month-by-month breakdowns of total customers, churned, and retained. Visualised as interactive line and bar charts.

![Monthly Trends Report](static/screenshots/monthly_trends_1779190432010.png)

---

## 🤖 Models Trained

| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| Decision Tree | ~99.8% | ~99.8% | ~99.8% |
| Logistic Regression | ~65% | ~65% | ~70% |
| Random Forest | ~99.9% | ~99.9% | ~99.9% |
| **XGBoost** ✅ | **~99.96%** | **~99.96%** | **~99.96%** |

> **Best Model:** XGBoost — deployed as the primary prediction engine.

---

## 🗂️ Project Structure

```
churnnn/
├── server.py                    # Flask backend (API + SPA serving)
├── train.py                     # Model training pipeline
├── utils.py                     # Helper functions
├── start.py                     # One-click launcher
├── requirements.txt             # Python dependencies
├── sample_bulk_prediction.csv   # Example bulk upload file
├── data/                        # Training dataset
├── models/                      # Saved ML models (.pkl)
├── static/
│   └── screenshots/             # Application screenshots
└── templates/                   # HTML templates
```

---

*Screenshots captured from the live running application at http://localhost:5000/*