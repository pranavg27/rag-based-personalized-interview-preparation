"""Streamlit entry point: Login / Register + landing page.  Run: streamlit run frontend/app.py"""
import streamlit as st

from components.ui import setup_page
from services.api_client import APIClient, APIError

setup_page("Home", "🎯")
st.title("🎯 RAG-Based Personalized Interview Preparation")
st.caption("Upload your resume and study material, practise adaptive mock interviews, get graded feedback.")

if st.session_state.get("token"):
    st.success(f"Signed in as **{st.session_state['user']['full_name']}**. Use the sidebar to navigate.")
    st.page_link("pages/1_Dashboard.py", label="Open dashboard", icon="📊")
    st.page_link("pages/5_Interview_Setup.py", label="Start an interview", icon="🎤")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    st.stop()

login_tab, register_tab = st.tabs(["Login", "Register"])
with login_tab:
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            try:
                data = APIClient().login(email, password)
                st.session_state.update(token=data["access_token"], user=data["user"])
                st.rerun()
            except APIError as exc:
                st.error(exc.detail)
with register_tab:
    with st.form("register"):
        name = st.text_input("Full name")
        email_r = st.text_input("Email", key="reg_email")
        pw = st.text_input("Password (8+ chars, letters and digits)", type="password", key="reg_pw")
        if st.form_submit_button("Create account"):
            try:
                APIClient().register(email_r, name, pw)
                data = APIClient().login(email_r, pw)
                st.session_state.update(token=data["access_token"], user=data["user"])
                st.rerun()
            except APIError as exc:
                st.error(exc.detail)
