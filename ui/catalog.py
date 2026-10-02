"""Scene catalog presentation and selection controls."""
from __future__ import annotations
import streamlit as st
from ui.layout import section_header, status_badge


def scene_label(item) -> str:
    date_str = str(item.datetime.date()) if item.datetime else "Unknown"
    cloud = float(item.properties.get("eo:cloud_cover", 0))
    return f"{date_str} · {cloud:.2f}% · {item.id[:14]}"


def render_scene_catalog(items, active_scene_id: str | None = None):
    section_header("Scenes", f"{len(items)} observations")
    if not items:
        st.caption("No scenes match the current filters.")
        return None

    header = st.columns([2.2, 1.2, 1.1, 1.0])
    for col, label in zip(header, ["Acquisition", "Cloud", "Status", "Action"]):
        with col:
            st.markdown(f'<div class="table-header">{label}</div>', unsafe_allow_html=True)

    selected = None
    for idx, item in enumerate(items):
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        ready = cloud <= 10
        row = st.columns([2.2, 1.2, 1.1, 1.0])
        with row[0]:
            st.markdown(f'<div class="table-cell"><strong>{date_str}</strong><span class="scene-id">{item.id}</span></div>', unsafe_allow_html=True)
        with row[1]:
            st.markdown(f'<div class="table-cell">{cloud:.2f}%</div>', unsafe_allow_html=True)
        with row[2]:
            st.markdown(f'<div class="table-cell">{status_badge("Ready" if ready else "Review", "ready" if ready else "pending")}</div>', unsafe_allow_html=True)
        with row[3]:
            if st.button("Select", key=f"scene_select_{idx}", use_container_width=True):
                selected = item
        if active_scene_id == item.id:
            st.markdown('<div class="active-scene-line">ACTIVE SCENE</div>', unsafe_allow_html=True)
    return selected
