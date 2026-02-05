import streamlit as st
from ui_config import hide_streamlit_nav
hide_streamlit_nav()


from sidebar import sidebar_menu


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("app.py")

sidebar_menu()


st.title("ℹ️ About the Project")

st.markdown("""
### 🎓 Project Title  
**Customer Churn Prediction Using Machine Learning**

### 🎯 Objective  
To predict customers who are likely to leave a service and help  
organizations take **data-driven retention decisions**.

### 🛠 Tech Stack  
- Python  
- Streamlit  
- Scikit-Learn  
- Random Forest  
- Pandas & NumPy  

### 🚀 Outcome  
A professional, AI-powered web application that assists businesses  
in reducing customer churn.
""")

st.image(
    "https://images.unsplash.com/photo-1521737604893-d14cc237f11d",
    use_container_width=True
)