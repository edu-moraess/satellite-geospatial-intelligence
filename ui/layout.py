from __future__ import annotations

import streamlit as st


def render_header(
    title: str = "Satellite Geospatial Intelligence",
    subtitle: str = "Earth Observation • Remote Sensing • Geospatial AI",
) -> None:
    """
    Render the main application header.
    """

    st.markdown(
        f"""
        <div class="sgi-header">

            <div class="sgi-brand">

                <div class="sgi-logo">
                    🛰️
                </div>

                <div>

                    <div class="sgi-title">
                        {title}
                    </div>

                    <div class="sgi-subtitle">
                        {subtitle}
                    </div>

                </div>

            </div>

            <div class="sgi-status">

                <span class="sgi-status-dot"></span>

                SYSTEM ONLINE

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(
    title: str,
    description: str | None = None,
    icon: str | None = None,
) -> None:
    """
    Render a compact section heading.
    """

    icon_html = (
        f"<span>{icon}</span>"
        if icon
        else ""
    )

    description_html = (
        f"""
        <span class="sgi-section-description">
            {description}
        </span>
        """
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="sgi-section">

            {icon_html}

            <span class="sgi-section-title">
                {title}
            </span>

            {description_html}

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_operations_header(
    title: str = "Geospatial Operations Center",
    meta: str = "Interactive Earth observation • Sentinel-2 • AOI • Spatial analysis",
) -> None:
    """
    Render the compact header above the main geospatial map.
    """

    st.markdown(
        f"""
        <div class="sgi-operations-title">

            <div class="sgi-operations-name">
                {title}
            </div>

            <div class="sgi-operations-meta">
                {meta}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """
    Render application footer.
    """

    st.markdown(
        """
        <div class="sgi-footer">

            SATELLITE GEOSPATIAL INTELLIGENCE
            <br>
            Earth Observation • Remote Sensing •
            Computer Vision • Geospatial AI

        </div>
        """,
        unsafe_allow_html=True,
    )