from __future__ import annotations

import streamlit as st

from services.api_client import APIClient


def get_api() -> APIClient:
    return APIClient(st.session_state.get("token"))


def require_login() -> APIClient:
    """Call at the top of every page. Stops rendering if the user is not signed in."""
    if not st.session_state.get("token"):
        st.warning("Please sign in on the home page first.")
        st.page_link("app.py", label="Go to Login / Register", icon="🔐")
        st.stop()
    with st.sidebar:
        user = st.session_state.get("user", {})
        st.markdown(f"**{user.get('full_name', '')}**  \n<span class='muted'>{user.get('email', '')}</span>",
                    unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    return get_api()
