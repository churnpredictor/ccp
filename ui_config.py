import streamlit as st

def hide_streamlit_nav():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSidebarNav"] + div {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)