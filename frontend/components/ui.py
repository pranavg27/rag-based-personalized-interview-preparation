from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.score-pill {display:inline-block; padding:.25rem .8rem; border-radius:999px; font-weight:700; color:white;}
.q-card {border:1px solid rgba(128,128,128,.35); border-left:5px solid #4f46e5; border-radius:8px; padding:1rem 1.2rem; margin:.5rem 0;}
.muted {opacity:.7; font-size:.9rem;}
</style>
"""


def setup_page(title: str, icon: str = "🎯") -> None:
    st.set_page_config(page_title=f"{title} · Interview Prep", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def score_pill(score: float | None, max_score: float = 10) -> str:
    if score is None:
        return "<span class='score-pill' style='background:#6b7280'>–</span>"
    color = "#16a34a" if score >= 7.5 else "#d97706" if score >= 5 else "#dc2626"
    return f"<span class='score-pill' style='background:{color}'>{score:.1f}/{max_score:.0f}</span>"


def show_error(exc: Exception) -> None:
    st.error(str(exc))
