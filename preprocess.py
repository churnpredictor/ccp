import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv('Customer_Churn.csv')  # update file name if needed

# 1. Remove duplicate records
df.drop_duplicates(inplace=True)

# 2. Fix incorrect data types
numeric_columns = [
    'Age',
    'Tenure',
    'Usage Frequency',
    'Support Calls',
    'Payment Delay',
    'Total Spend',
    'Last Interaction'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Fix target variable
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].astype(int)

# 4. Handle missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. Drop irrelevant column
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# 6. Encode categorical variables
le = LabelEncoder()
categorical_features = ['Gender', 'Subscription Type', 'Contract Length']

for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# 7. Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 8. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data preprocessing completed successfully")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
