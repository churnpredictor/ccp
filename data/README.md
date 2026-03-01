# Dataset Download Instructions

## 📥 Download the Telecom Churn Dataset

This project requires the **Telco Customer Churn** dataset from Kaggle.

### Option 1: Manual Download (Recommended)

1. **Visit Kaggle:**
   - Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. **Sign In:**
   - Create a Kaggle account if you don't have one
   - Sign in to your account

3. **Download:**
   - Click the "Download" button
   - The file `WA_Fn-UseC_-Telco-Customer-Churn.csv` will download

4. **Place in Project:**
   - Move the downloaded CSV file to the `data/` folder in this project
   - The full path should be: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Option 2: Using Kaggle API (Advanced)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (follow Kaggle API documentation)
# Download dataset
kaggle datasets download -d blastchar/telco-customer-churn

# Unzip
unzip telco-customer-churn.zip -d data/
```

### Dataset Information

- **File Name:** WA_Fn-UseC_-Telco-Customer-Churn.csv
- **Size:** ~1 MB
- **Records:** ~7,000 customers
- **Features:** 21 columns

**Columns:**
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn (Target Variable)

### Verification

After placing the file, verify with:

```bash
# Windows
dir data\WA_Fn-UseC_-Telco-Customer-Churn.csv

# Linux/Mac
ls -lh data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Next Steps

Once the dataset is in place:

1. Train the models:
   ```bash
   python train.py
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

---

**Important:** The dataset is required before running `train.py`. The training script will not work without it.
