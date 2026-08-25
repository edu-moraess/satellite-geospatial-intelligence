from __future__ import annotations

import streamlit as st


def metric_card(
    label: str,
    value: str,
    description: str | None = None,
) -> None:
    """
    Render a professional metric card.
    """

    description_html = (
        f"""
        <div class="sgi-card-description">
            {description}
        </div>
        """
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="sgi-card">

            <div class="sgi-card-label">
                {label}
            </div>

            <div class="sgi-card-value">
                {value}
            </div>

            {description_html}

        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    description: str | None = None,
) -> None:
    """
    Compact KPI card used in the dashboard header.
    """

    description_html = (
        f"""
        <div class="sgi-kpi-sub">
            {description}
        </div>
        """
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="sgi-kpi">

            <div class="sgi-kpi-label">
                {label}
            </div>

            <div class="sgi-kpi-value">
                {value}
            </div>

            {description_html}

        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(
    label: str,
    status: str = "online",
) -> None:
    """
    Render a small system status badge.
    """

    status_map = {
        "online": ("#6FD39A", "ONLINE"),
        "warning": ("#D6B86A", "WARNING"),
        "error": ("#D97979", "ERROR"),
        "offline": ("#7C8A90", "OFFLINE"),
    }

    dot_color, text = status_map.get(
        status,
        status_map["offline"],
    )

    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:0.45rem;
            padding:0.35rem 0.65rem;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.08);
            background:rgba(255,255,255,0.025);
            font-size:0.68rem;
            color:#AAB8C0;
        ">

            <span style="
                width:6px;
                height:6px;
                border-radius:50%;
                background:{dot_color};
            "></span>

            {label.upper()} · {text}

        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(
    title: str,
    value: str,
    description: str | None = None,
) -> None:
    """
    Render a generic information card.
    """

    description_html = (
        f"""
        <div style="
            color:#71838C;
            font-size:0.68rem;
            margin-top:0.35rem;
        ">
            {description}
        </div>
        """
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="sgi-card">

            <div class="sgi-card-label">
                {title}
            </div>

            <div style="
                font-size:0.95rem;
                font-weight:600;
                color:#E8EEF2;
            ">
                {value}
            </div>

            {description_html}

        </div>
        """,
        unsafe_allow_html=True,
    )