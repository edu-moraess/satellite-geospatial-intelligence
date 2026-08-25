"""
ui/theme.py — Minimal technical CSS for Satellite Geospatial Intelligence.

Does NOT override Streamlit theme background or global text color.
Light/Dark mode remains under Streamlit control.
CSS is limited to spacing, borders, typography, and component sizing.
"""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Inject restrained workstation-style CSS. Theme-safe (no forced backgrounds)."""
    st.markdown(
        """
<style>
/* ---- Layout density ---- */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1280px;
}

/* ---- Header bar ---- */
.sgi-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.35rem 0 0.55rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.28);
    flex-wrap: wrap;
}
.sgi-brand {
    display: flex;
    align-items: baseline;
    gap: 0.65rem;
    min-width: 0;
}
.sgi-mark {
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: inherit;
    opacity: 0.85;
}
.sgi-title {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.2;
}
.sgi-subtitle {
    font-size: 0.72rem;
    opacity: 0.65;
    margin-top: 0.1rem;
    letter-spacing: 0.01em;
}
.sgi-system {
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    opacity: 0.8;
    white-space: nowrap;
}
.sgi-system .dot {
    display: inline-block;
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 50%;
    background: #22a06b;
    margin-right: 0.35rem;
    vertical-align: middle;
}

/* ---- Section headers ---- */
.sgi-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.7;
    margin: 0.9rem 0 0.15rem 0;
}
.sgi-section-desc {
    font-size: 0.78rem;
    opacity: 0.6;
    margin: 0 0 0.55rem 0;
}

/* ---- Metric cards (technical) ---- */
.sgi-metric {
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-radius: 4px;
    padding: 0.45rem 0.6rem;
    margin: 0.15rem 0;
}
.sgi-metric .label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.6;
}
.sgi-metric .value {
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.35;
    margin-top: 0.1rem;
}
.sgi-metric .hint {
    font-size: 0.68rem;
    opacity: 0.55;
    margin-top: 0.05rem;
}

/* ---- Pipeline status ---- */
.sgi-pipe {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1.25rem;
    padding: 0.45rem 0;
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}
.sgi-pipe .stage {
    opacity: 0.9;
}
.sgi-pipe .stage .dot {
    display: inline-block;
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 50%;
    margin-right: 0.35rem;
    vertical-align: middle;
}
.sgi-pipe .ready .dot { background: #22a06b; }
.sgi-pipe .pending .dot { background: rgba(128, 128, 128, 0.55); }
.sgi-pipe .active .dot { background: #e2a03f; }
.sgi-pipe .error .dot { background: #c9372c; }

/* ---- Mono helpers for technical data ---- */
.sgi-mono {
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.82em;
}

/* ---- Footer ---- */
.sgi-footer {
    margin-top: 1.25rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    font-size: 0.68rem;
    opacity: 0.55;
    text-align: center;
    line-height: 1.45;
}

/* ---- Streamlit density tweaks (no color overrides) ---- */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 4px;
    padding: 0.35rem 0.55rem;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 4px;
}
.stAlert {
    padding: 0.4rem 0.7rem;
    font-size: 0.85rem;
}

@media (max-width: 768px) {
    .sgi-header { gap: 0.4rem; }
    .sgi-title { font-size: 0.88rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )
