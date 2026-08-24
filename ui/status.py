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
    Render the current state of the geospatial pipeline.
    """

    st.markdown(
        "### Processing Pipeline"
    )

    stages = [
        ("Catalog", scene_available),
        ("Imagery", imagery_available),
        ("Spectral", spectral_available),
        ("Change Detection", change_available),
        ("Geospatial AI", ai_available),
    ]

    columns = st.columns(
        len(stages)
    )

    for column, (name, available) in zip(
        columns,
        stages,
    ):

        with column:

            if available:

                st.success(
                    f"● {name}",
                    icon="✓",
                )

            else:

                st.info(
                    f"○ {name}",
                )