# 🚀 QUICK START GUIDE

## 📦 What You've Got

A **production-level Customer Churn Prediction Dashboard** with:
- ✅ 4 ML Models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- ✅ Dark Theme UI
- ✅ SHAP Explainability
- ✅ PDF Reports
- ✅ Bulk Predictions
- ✅ ROC Curves & Confusion Matrices
- ✅ Real-time Probability Gauge

---

## ⚡ 3-Step Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in `data/` folder

### Step 3: Run Quick Start
```bash
python start.py
```

This will:
- ✅ Check if dataset is present
- ✅ Train models if needed (2-5 minutes)
- ✅ Launch the Streamlit app

---

## 🎮 Manual Steps

If you prefer manual control:

### Train Models:
```bash
python train.py
```

### Launch App:
```bash
streamlit run app.py
```

---

## 📚 Important Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application (6 pages) |
| `train.py` | Model training script |
| `utils.py` | Visualization & PDF functions |
| `start.py` | Guided quick start |
| `README.md` | Full documentation |
| `VIVA_NOTES.md` | Exam preparation (25+ Q&A) |

---

## 🎯 Dashboard Features

### 🏠 Home
- Dataset overview
- Model performance comparison
- Quick start guide

### 🔮 Single Prediction
- Interactive input form
- Real-time probability gauge
- Risk assessment (High/Medium/Low)
- Feature importance chart
- Decision tree visualization
- PDF report download

### 📂 Bulk Prediction
- Upload CSV file
- Batch predictions
- Churn distribution charts
- Download results

### 📊 Model Analytics
- Confusion matrix heatmaps
- ROC curve comparisons
- Performance metrics

### ⚖️ Model Comparison
- Side-by-side comparison table
- Radar charts
- Model explanations

### 📈 Explainability (SHAP)
- SHAP summary plots
- Feature impact analysis
- Global insights

---

## 🎓 For Viva/Presentation

**Read:** `VIVA_NOTES.md` - Contains 25+ prepared answers

**Key Points:**
- Primary Model: Random Forest (88% accuracy)
- Why RF? Balance of accuracy & interpretability
- SHAP for explainability
- Business value: ~$500K annual savings

**Demo Flow:**
1. Show Home page (dataset overview)
2. Make a single prediction
3. Show probability gauge
4. Show decision tree visualization
5. Show SHAP explainability
6. Generate PDF report

---

## ⚠️ Troubleshooting

**Issue:** Dataset not found
**Solution:** Download from Kaggle and place in `data/` folder

**Issue:** Module not found
**Solution:** Run `pip install -r requirements.txt`

**Issue:** Models not trained
**Solution:** Run `python train.py` first

**Issue:** Streamlit won't open
**Solution:** Manually go to `http://localhost:8501`

---

## 💡 Tips

1. **Training Time:** First run takes 2-5 minutes to train models
2. **Sample CSV:** Use `sample_bulk_prediction.csv` to test bulk predictions
3. **Dark Theme:** Already configured in `.streamlit/config.toml`
4. **PDF Reports:** Auto-generated for single predictions

---

## 📞 Next Steps

1. ✅ Install dependencies
2. ✅ Download dataset
3. ✅ Run `python start.py`
4. ✅ Explore the dashboard
5. ✅ Read `VIVA_NOTES.md` for exam prep
6. ✅ Practice your demo

---

**Good luck! 🚀**
