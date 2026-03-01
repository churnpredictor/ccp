"""
Model Training Script for Customer Churn Prediction
Trains multiple models: Logistic Regression, Decision Tree, Random Forest, XGBoost
Adapted for the customer churn dataset with 440K+ records
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 80)

# Load dataset
print("\n[1/8] Loading dataset...")
try:
    df = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    print(f"✓ Dataset loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Dataset not found!")
    print("\nPlease ensure 'customer_churn_dataset-training-master.csv' is in the data/ folder")
    exit(1)

# Data Preprocessing
print("\n[2/8] Preprocessing data...")

# Display initial info
print(f"✓ Total records: {len(df):,}")
print(f"✓ Columns: {df.columns.tolist()}")

# Drop CustomerID as it's not a feature
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# Handle missing values
missing_before = df.isnull().sum().sum()
if missing_before > 0:
    print(f"✓ Found {missing_before} missing values, handling them...")
    # Drop rows with missing values for simplicity (you could also impute)
    df = df.dropna()
    print(f"✓ After dropping nulls: {len(df):,} rows remaining")

# Identify target variable
if 'Churn' in df.columns:
    target_col = 'Churn'
else:
    print("✗ 'Churn' column not found!")
    exit(1)

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Convert target to binary if needed
if y.dtype == 'float64' or y.dtype == 'int64':
    # Already numeric
    y = y.astype(int)
else:
    # Encode if categorical
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print(f"✓ Churn distribution: {pd.Series(y).value_counts().to_dict()}")
print(f"✓ Churn rate: {(y.sum() / len(y) * 100):.2f}%")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"✓ Categorical features: {len(categorical_cols)} - {categorical_cols}")
print(f"✓ Numerical features: {len(numerical_cols)} - {numerical_cols}")

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, 'models/label_encoders.pkl')
print("✓ Label encoders saved")

# Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Scaler saved")

# Save feature columns
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("✓ Feature columns saved")

# Train-test split (use smaller sample for faster training if dataset is huge)
print("\n[3/8] Splitting data...")
# For very large datasets, we can sample for faster training
if len(X_scaled) > 100000:
    print(f"✓ Large dataset detected ({len(X_scaled):,} records)")
    print("✓ Using stratified sample of 50,000 records for training")
    from sklearn.model_selection import train_test_split as sample_split
    X_sample, _, y_sample, _ = sample_split(
        X_scaled, y, train_size=50000, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# Initialize models
print("\n[4/8] Initializing models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, min_samples_split=10, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1, eval_metric='logloss', n_jobs=-1)
}
print("✓ Models initialized")

# Train and evaluate models
print("\n[5/8] Training models...")
results = {}
predictions = {}
probabilities = {}
roc_data = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store predictions and probabilities
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Store ROC data
    roc_data[name] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1 Score: {f1:.4f}")
    print(f"  ✓ AUC: {roc_auc:.4f}")

# Save models
print("\n[6/8] Saving models...")
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

for name, model in models.items():
    filename = f'models/{model_filenames[name]}'
    joblib.dump(model, filename)
    print(f"✓ {name} saved to {filename}")

# Save evaluation results
print("\n[7/8] Saving evaluation metrics...")
joblib.dump(results, 'models/evaluation_results.pkl')
joblib.dump(roc_data, 'models/roc_data.pkl')
print("✓ Evaluation metrics saved")

# Save test data for later use
print("\n[8/8] Saving test data...")
test_data = {
    'X_test': X_test,
    'y_test': y_test
}
joblib.dump(test_data, 'models/test_data.pkl')
print("✓ Test data saved")

# Display summary
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"\nDataset: {len(df):,} total records")
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Features: {len(feature_columns)}")
print(f"Churn rate: {(y.sum() / len(y) * 100):.2f}%")

print("\n" + "-" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("-" * 80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'AUC':<12}")
print("-" * 80)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['auc']:<12.4f}")

print("-" * 80)

# Identify best model
best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
print(f"\n🏆 Best Model: {best_model[0]} (F1 Score: {best_model[1]['f1_score']:.4f})")

print("\n" + "=" * 80)
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nAll models and data have been saved to the 'models/' directory.")
print("You can now run the Streamlit app with: streamlit run app.py")
print("\n" + "=" * 80)
