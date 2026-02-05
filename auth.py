import streamlit as st
import json
import hashlib
from pathlib import Path

USER_FILE = Path("users.json")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if USER_FILE.exists():
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register():
    st.subheader("📝 Create New Account")

    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    confirm = st.text_input("🔑 Confirm Password", type="password")

    if st.button("Register"):
        users = load_users()

        if not email or not password:
            st.error("All fields are required")
        elif password != confirm:
            st.error("Passwords do not match")
        elif email in users:
            st.error("Email already registered")
        else:
            users[email] = hash_password(password)
            save_users(users)
            st.success("🎉 Registration successful! Please login.")

def login():
    st.subheader("🔐 Login")

    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        users = load_users()

        if email in users and users[email] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.user = email
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid email or password")

def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()