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

## 🖼️ Application Screenshots

### 1. 📊 Dashboard — Overview & KPIs

The main dashboard displays live KPIs: total customers, active vs. churned counts, overall churn rate, and a model performance comparison table.

<img width="1917" height="969" alt="Screenshot 2026-05-19 160108" src="https://github.com/user-attachments/assets/443264fc-bb20-40a2-9f02-0402c0f747bb" />

<img width="1910" height="922" alt="Screenshot 2026-05-19 160134" src="https://github.com/user-attachments/assets/03363879-93e3-4556-af9e-970f2c2747c4" />

---

### 2. 👥 Customer History & Search

Search and browse churned customer records. Filter by date range or search by Customer ID / Name to retrieve full profiles.

<img width="1919" height="963" alt="Screenshot 2026-05-19 160201" src="https://github.com/user-attachments/assets/a31bb867-ed15-4375-8634-fa2c873189fa" />
<img width="1919" height="958" alt="Screenshot 2026-05-19 160332" src="https://github.com/user-attachments/assets/5a9a6b01-7db1-46cc-85f5-6e2a86cb609f" />



---

### 3. 🔮 Single Prediction

Enter individual customer details to get a real-time churn probability score, risk level classification, and SHAP-powered explainability reasons.
<img width="1919" height="976" alt="Screenshot 2026-05-19 160229" src="https://github.com/user-attachments/assets/914dad31-71ad-41da-8042-201db583cbc4" />
<img width="1918" height="915" alt="Screenshot 2026-05-19 160504" src="https://github.com/user-attachments/assets/195e2daa-d01a-4b75-a2d7-c9654326db71" />



---

### 4. 📂 Bulk Prediction

Upload a CSV file to run batch predictions across hundreds or thousands of customers at once, with cohort risk distribution charts.

<img width="1919" height="962" alt="Screenshot 2026-05-19 160814" src="https://github.com/user-attachments/assets/6177f1ab-4d24-4a08-9fe3-f383bc35f588" />

<img width="1919" height="970" alt="Screenshot 2026-05-19 160900" src="https://github.com/user-attachments/assets/133c38e1-ffa8-4a91-88ca-314176a775b1" />

---

### 5. 📈 Model Analytics

Deep-dive into ML model performance metrics — interactive ROC curves, confusion matrices, and feature importance rankings for all 4 trained models.


---

### 6. 🧠 SHAP Explainability

View SHAP (SHapley Additive exPlanations) feature importance values for Random Forest and XGBoost, revealing which features drive churn predictions the most.

![SHAP Explainability](static/screenshots/06_explainability.png)

---

### 7. 🌳 Decision Tree Prediction

Visualise exactly how the Decision Tree model makes its churn decision, step-by-step — with interactive sliders for each customer feature.

![Decision Tree Prediction](static/screenshots/07_decision_tree.png)

---

### 8. ⚖️ Model Comparison

Side-by-side comparison of all 4 trained models using a radar chart spanning Accuracy, Precision, Recall, F1 Score, and AUC — plus individual model detail cards.

![Model Comparison](static/screenshots/08_model_comparison.png)

---

### 9. 📅 Monthly Trends

Month-wise churn trend analysis from 2021 to 2026 — stacked bar charts showing churned vs. retained customers per month, plus a churn rate trend line.

![Monthly Trends](static/screenshots/09_monthly_trends.png)

---

<div align="center">
  <b>Dataset by Muhammad Shahid Azeem on Kaggle</b><br>
  <a href="https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset">🔗 https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset</a>
</div>
