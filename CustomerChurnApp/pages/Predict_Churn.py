import streamlit as st
import pandas as pd
import joblib

from ui_config import hide_streamlit_nav
hide_streamlit_nav()

from sidebar import sidebar_menu

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("app.py")

sidebar_menu()

# 🔒 Protect page
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("🔒 Please login to access this page.")
    st.stop()

st.title("🔮 Customer Churn Prediction")
st.markdown("### AI-powered churn risk assessment")

# Load model & features
model = joblib.load("churn_model_xgb.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.markdown("#### 🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    usage = st.slider("Usage Frequency", 1, 30, 15)
    support = st.slider("Support Calls", 0, 10, 2)

with col2:
    payment_delay = st.slider("Payment Delay (Days)", 0, 30, 5)
    total_spend = st.number_input("Total Spend ($)", 100, 10000, 800)
    last_interaction = st.slider("Last Interaction (Days)", 1, 60, 10)

with col3:
    gender = st.selectbox("Gender", ["Female", "Male"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Annual", "Quarterly", "Monthly"])

if st.button("🚀 Predict Churn Risk"):
    # Base input
    input_data = {
        "Age": age,
        "Tenure": tenure,
        "UsageFrequency": usage,
        "SupportCalls": support,
        "PaymentDelay": payment_delay,
        "TotalSpend": total_spend,
        "LastInteraction": last_interaction,
        "Gender_Male": 1 if gender == "Male" else 0,
        "SubscriptionType_Premium": 1 if subscription == "Premium" else 0,
        "SubscriptionType_Standard": 1 if subscription == "Standard" else 0,
        "ContractLength_Monthly": 1 if contract == "Monthly" else 0,
        "ContractLength_Quarterly": 1 if contract == "Quarterly" else 0
    }

    # Create DataFrame with EXACT columns
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Churn Risk ({1 - probability:.2%})")