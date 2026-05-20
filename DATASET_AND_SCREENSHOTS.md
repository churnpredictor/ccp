
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

The main dashboard displays live KPIs: total customers, active vs. churned counts, overall churn rate

<img width="1919" height="920" alt="image" src="https://github.com/user-attachments/assets/9b5f8d85-181b-4792-82d2-e1cf79045b48" />


---

### 2. 👥 Customer History & Search

Search and browse churned customer records. Filter by
1.Date range
<img width="1905" height="974" alt="image" src="https://github.com/user-attachments/assets/eef0bd58-1d0a-4738-85fd-63d6517d5486" />
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/a68e6b2d-53d3-42c0-9d67-be2f5077f57a" />



2.Search by Customer ID / Name to retrieve full profiles.

<img width="1911" height="955" alt="image" src="https://github.com/user-attachments/assets/58b64759-29ce-4980-85be-2209dfa5c9cb" />
<img width="1919" height="921" alt="image" src="https://github.com/user-attachments/assets/56458975-1bba-45c5-8732-b70686b98efe" />
<img width="1919" height="968" alt="image" src="https://github.com/user-attachments/assets/0a1708c5-314e-4a47-b003-7712323e720b" />





---

### 3. 🔮 Single Prediction

Enter individual customer details to get a real-time churn probability score, risk level classification, and SHAP-powered explainability reasons.
<img width="1916" height="1044" alt="image" src="https://github.com/user-attachments/assets/6705dea4-93fd-4134-a3c1-bac8f2a16780" />

<img width="1919" height="1011" alt="image" src="https://github.com/user-attachments/assets/5430580e-c5bd-4064-a570-683d02a86d7d" />

<img width="1919" height="1042" alt="image" src="https://github.com/user-attachments/assets/c5d3be2b-c8e3-4364-a618-57bd8cef53f8" />

---

### 4. 📂 Bulk Prediction

Upload a CSV file to run batch predictions across hundreds or thousands of customers at once, with cohort risk distribution charts.

<img width="1919" height="986" alt="image" src="https://github.com/user-attachments/assets/c39729d6-2a59-42e7-a6b1-4d453c1258c7" />
<img width="1919" height="996" alt="image" src="https://github.com/user-attachments/assets/b14016d6-d5fc-416f-8324-8176b89cc1b6" />
<img width="1919" height="1013" alt="image" src="https://github.com/user-attachments/assets/6217668a-11fd-40bf-8047-20bfb9ae8fcd" />

---

### 5. 📈 Model Analytics

Deep-dive into ML model performance metrics —  confusion matrices, and feature importance rankings for all 4 trained models.
1.Confusion matrices
<img width="1919" height="990" alt="image" src="https://github.com/user-attachments/assets/5b2eafd1-e6a6-4f99-971d-82f77a913e06" />
<img width="1919" height="917" alt="image" src="https://github.com/user-attachments/assets/700334b0-8333-4b94-8809-72467de9304c" />
<img width="1918" height="975" alt="image" src="https://github.com/user-attachments/assets/acef0e2a-ec83-4463-b6ca-9ed100fb0c06" />
<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/ba608a0f-f884-44ea-a36f-e80e08662270" />
2.Perfomance
<img width="1919" height="949" alt="image" src="https://github.com/user-attachments/assets/8778b461-bad6-4bd3-b63b-6fd44806a218" />


---

### 6. 🧠 SHAP Explainability

View SHAP (SHapley Additive exPlanations) feature importance values for Random Forest and XGBoost, revealing which features drive churn predictions the most.

<img width="1919" height="984" alt="image" src="https://github.com/user-attachments/assets/3dc06ffd-965e-4a63-97e3-e56ee8434f1c" />
<img width="1907" height="950" alt="image" src="https://github.com/user-attachments/assets/82518e5b-9ee2-4f37-a907-2ed662b7b139" />



---

### 7. 🌳 Decision Tree Prediction

Visualise exactly how the Decision Tree model makes its churn decision, step-by-step — with interactive sliders for each customer feature.

<img width="1914" height="970" alt="image" src="https://github.com/user-attachments/assets/bbdd3b80-c817-47f1-af35-26e93e8a20a2" />
<img width="1911" height="884" alt="image" src="https://github.com/user-attachments/assets/8545ba41-fa61-4017-bc4c-40d43117b69b" />
<img width="1919" height="955" alt="image" src="https://github.com/user-attachments/assets/8161d818-c46f-49c6-880c-44bbc3c2fbc0" />
<img width="1919" height="941" alt="image" src="https://github.com/user-attachments/assets/68481c19-4fe4-42c6-a20a-8fa4a272d1c3" />


---

### 8. ⚖️ Model Comparison

Side-by-side comparison of all 4 trained models using a radar chart spanning Accuracy, Precision, Recall, F1 Score, and AUC — plus individual model detail cards.

<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/8fd7b8c4-9f5b-463e-afd5-bf1fa77f61a5" />

<img width="1919" height="901" alt="image" src="https://github.com/user-attachments/assets/ea979231-03b2-4095-80a1-166106c85250" />


---


<div align="center">
  <b>Dataset by Muhammad Shahid Azeem on Kaggle</b><br>
  <a href="https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset">🔗 https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset</a>
</div>
