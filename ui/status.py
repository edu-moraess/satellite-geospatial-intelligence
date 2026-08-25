from __future__ import annotations

import streamlit as st


def render_pipeline_status(
    scene_available: bool = False,
    imagery_available: bool = False,
    spectral_available: bool = False,
    change_available: bool = False,
    ai_available: bool = False,
) -> None:
    """
    Render a compact horizontal geospatial processing pipeline.
    """

    stages = [
        ("Catalog", scene_available),
        ("Imagery", imagery_available),
        ("Spectral", spectral_available),
        ("Change Detection", change_available),
        ("Geospatial AI", ai_available),
    ]

    parts: list[str] = []

    for index, (name, available) in enumerate(stages):

        active_class = (
            "active"
            if available
            else ""
        )

        symbol = "✓" if available else "○"

        parts.append(
            f"""
            <div class="sgi-pipeline-stage {active_class}">
                <span>{symbol}</span>
                <span>{name}</span>
            </div>
            """
        )

        if index < len(stages) - 1:

            parts.append(
                """
                <span class="sgi-pipeline-arrow">
                    →
                </span>
                """
            )

    st.markdown(
        f"""
        <div class="sgi-pipeline">
            {''.join(parts)}
        </div>
        """,
        unsafe_allow_html=True,
    )