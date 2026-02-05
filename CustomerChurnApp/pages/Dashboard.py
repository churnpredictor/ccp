import streamlit as st
import pandas as pd
import joblib
from ui_config import hide_streamlit_nav
hide_streamlit_nav()

from sidebar import sidebar_menu

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("app.py")

sidebar_menu()


st.title("📊 Customer Churn Analytics Dashboard")
st.markdown("### Real-time business insights & retention metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Customers", "64,374")
col2.metric("📉 Churn Rate", "28%")
col3.metric("📈 Retention Rate", "72%")
col4.metric("💰 Avg Revenue", "$1,240")

st.markdown("---")

left, right = st.columns([2, 1])

with left:
    st.subheader("📌 Business Overview")
    st.markdown("""
    - Monitor customer behavior patterns  
    - Identify churn-prone segments  
    - Improve retention strategies  
    - Reduce revenue loss  
    """)

with right:
    st.image(
        "https://images.unsplash.com/photo-1556761175-4b46a572b786",
        use_container_width=True
    )

st.markdown("---")

st.info("💡 **Insight:** Customers with high support calls and payment delays are more likely to churn.")