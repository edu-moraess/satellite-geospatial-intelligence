"""
ui/status.py — Pipeline stage status in session_state.
"""

from __future__ import annotations

import streamlit as st

PIPELINE_STAGES = ("Catalog", "Imagery", "Spectral", "Change", "AI")


def init_pipeline_status() -> None:
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = {
            stage: "pending" for stage in PIPELINE_STAGES
        }


def get_pipeline_status() -> dict:
    init_pipeline_status()
    return st.session_state["pipeline_status"]


def update_pipeline_status(stage: str, state: str) -> None:
    """
    Update one pipeline stage.
    state: pending | active | done | error
    """
    init_pipeline_status()
    if stage in st.session_state["pipeline_status"]:
        st.session_state["pipeline_status"][stage] = state


def reset_pipeline() -> None:
    init_pipeline_status()
    for stage in PIPELINE_STAGES:
        st.session_state["pipeline_status"][stage] = "pending"
