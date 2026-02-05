import streamlit as st

def sidebar_menu():
    with st.sidebar:
        st.markdown("## 📊 ChurnGuard AI")
        st.caption("AI-Powered Customer Retention")

        st.markdown("---")
        st.markdown(f"👤 **{st.session_state.user_email}**")
        st.markdown("---")

        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/Dashboard.py", label="📊 Dashboard")
        st.page_link("pages/Predict_Churn.py", label="🔮 Predict Churn")
        st.page_link("pages/Model_Insights.py", label="📈 Model Insights")
        st.page_link("pages/About.py", label="ℹ️ About")

        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()