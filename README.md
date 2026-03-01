# 🎯 Smart Customer Churn Prediction System

## 🌟 Overview
The **Smart Customer Churn Prediction System** is a production-level analytical pipeline and web dashboard built with **Flask** and modern frontend technologies. It leverages an ensemble of Machine Learning models to predict customer churn in the telecommunications industry, providing highly actionable insights using **Explainable AI** (SHAP) and interactive **Plotly** visualizations.

---

## 🚀 Features

- **🔮 Single Customer Prediction**: Input specific customer details to receive real-time churn probability, complete with risk leveling (Low/Medium/High) and an interactive **Plotly** gauge chart.
- **📂 Bulk Prediction**: Upload customer datasets via CSV for high-throughput batch processing. Instantly track aggregate churn distribution and identify at-risk cohorts.
- **🧠 Explainable AI (SHAP)**: Understand exactly *why* a prediction was made. SHAP analysis generates human-readable explanations (e.g., "Short Tenure (2 months) increases churn risk") and presents tailored retention recommendations.
- **📊 Model Analytics & Comparison**: Dive deep into machine learning performance metrics. Visualize Confusion Matrices, interactive ROC curves across all 4 models, and dynamic Feature Importance ranking.
- **🌳 Decision Tree Visualization**: View the literal "If-Then" node pathways computed by the Decision Tree model to completely map the decision logic.
- **📈 Month-wise Trend Analysis**: Analyze influx of churn over selected monthly timeframes to observe broader system patterns.

---

## 🏗 System Architecture

### Technology Stack
- **Backend:** Python, Flask server (`server.py`)
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Plotly.js

### Models Leveraged
1. **Logistic Regression** - A reliable, fast, interpretable baseline model.
2. **Decision Tree** - Built for maximum transparency with node-by-node visual pathways.
3. **Random Forest (Primary Engine)** - High-accuracy ensemble engine achieving **88% Accuracy**. Reduces overfitting through bagging and provides exceptional feature importance tracking.
4. **XGBoost (Advanced)** - Gradient-boosting powerhouse used for achieving industry-standard maximum predictive performance. 

---

## 💻 Installation & Usage

### Step 1: Install Dependencies
Ensure you have Python 3.8+ installed on your system. Navigate to the project directory and install the required modules:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Flask Server
Run the main server application script to load the machine learning models and host the API:
```bash
python server.py
```

### Step 3: Access the Dashboard
By default, the application will initialize on port 5000.
Open your preferred web browser and navigate to:
```text
http://localhost:5000
```

---

## 📁 Project Structure

```text
churnnn/
│
├── data/                  # CSV datasets (e.g., customer_churn_dataset.csv)
├── models/                # Saved / Pre-trained .pkl models, encoders, and scalers
├── static/                # Frontend assets (CSS styling, icons)
├── templates/             # Frontend HTML (index.html)
│
├── server.py              # Main Flask application, routing, and API endpoints
├── train.py               # ML Training pipeline script (run to retrain models)
├── utils.py               # Data processing and preprocessing utilities
├── requirements.txt       # Python library dependencies
└── README.md              # Project documentation
```

---

## 👨‍💻 Note on Training
If you wish to retrain the models from scratch or test newly adjusted parameters, simply run:
```bash
python train.py
```
*This will parse the data inside the `data/` folder, train the 4 respective models, serialize them, and update the `.pkl` files inside the `models/` directory automatically.*
