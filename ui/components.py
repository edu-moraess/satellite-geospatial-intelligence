"""
ui/components.py — Reusable technical UI primitives.
"""

from __future__ import annotations

import streamlit as st


def metric_card(label: str, value: str, hint: str | None = None) -> None:
    """Compact technical metric (neutral border, mono value)."""
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
<div class="sgi-metric">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {hint_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_stage(name: str, state: str) -> None:
    """
    Render one pipeline stage line segment.
    state: pending | active | done | error  (done maps to ready)
    """
    css = {
        "done": "ready",
        "ready": "ready",
        "active": "active",
        "pending": "pending",
        "error": "error",
        "offline": "pending",
    }.get(state, "pending")
    label = {
        "done": "READY",
        "ready": "READY",
        "active": "ACTIVE",
        "pending": "PENDING",
        "error": "ERROR",
        "offline": "OFFLINE",
    }.get(state, state.upper())
    st.markdown(
        f'<span class="stage {css}"><span class="dot"></span>'
        f'{name.upper()}  {label}</span>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    """Compact section label + optional description."""
    st.markdown(f'<div class="sgi-section-label">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(
            f'<div class="sgi-section-desc">{description}</div>',
            unsafe_allow_html=True,
        )


def mono(text: str) -> str:
    """Wrap technical text for monospace display."""
    return f'<span class="sgi-mono">{text}</span>'
