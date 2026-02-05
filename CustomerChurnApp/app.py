import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# HIDE DEFAULT STREAMLIT PAGE NAVIGATION (IMPORTANT FIX)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit default multipage menu */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# --------------------------------------------------
# LOGIN & REGISTER UI
# --------------------------------------------------
def login_page():
    st.markdown("## 🔐 Login to **ChurnGuard AI**")
    st.write("Secure AI-powered customer churn prediction platform")

    with st.form("login_form"):
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if email and password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Please enter email and password")

    st.markdown("---")
    st.info("🆕 New user? Use Register option below")

    with st.form("register_form"):
        st.subheader("📝 Register")
        reg_email = st.text_input("📧 Register Email")
        reg_password = st.text_input("🔑 Create Password", type="password")
        reg_submit = st.form_submit_button("Register")

        if reg_submit:
            if reg_email and reg_password:
                st.success("🎉 Registration successful! Please login.")
            else:
                st.error("❌ All fields required")

# --------------------------------------------------
# SIDEBAR (ONLY AFTER LOGIN)
# --------------------------------------------------
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

# --------------------------------------------------
# MAIN HOME PAGE
# --------------------------------------------------
def home_page():
    st.markdown("## 🏠 Welcome to **ChurnGuard AI**")
    st.subheader("AI-Powered Customer Churn Prediction System")

    st.markdown(
        """
        **ChurnGuard AI helps organizations to:**
        - 🔍 Identify customers likely to leave
        - 📊 Understand churn-driving factors
        - 📈 Improve customer retention strategies
        - 💰 Reduce revenue loss
        """
    )

    st.success("👉 Use the sidebar to explore the application")

    st.image(
        "https://images.unsplash.com/photo-1556761175-5973dc0f32e7",
        caption="Data-driven customer insights",
        use_container_width=True
    )

# --------------------------------------------------
# APP FLOW
# --------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    sidebar_menu()
    home_page()