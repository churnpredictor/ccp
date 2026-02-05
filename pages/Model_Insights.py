import streamlit as st
import streamlit as st
from ui_config import hide_streamlit_nav
hide_streamlit_nav()


from sidebar import sidebar_menu


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.switch_page("app.py")

sidebar_menu()

st.title("🧠 Machine Learning Model Insights")

st.markdown("""
### 📌 Models Evaluated
- Logistic Regression  
- Decision Tree  
- **Random Forest (Final Model)**  
""")

st.success("""
🏆 **Random Forest Selected**
- Highest accuracy: **99.8%**
- Handles large datasets efficiently
- Reduces overfitting
- Strong generalization
""")

st.markdown("---")

st.subheader("📊 Accuracy Comparison")

st.markdown("""
| Model | Accuracy |
|------|---------|
| Logistic Regression | 83% |
| Decision Tree | 99.7% |
| **Random Forest** | **99.8%** |
""")

st.image(
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
    use_container_width=True
)