"""
ui/layout.py — Presentation layer for Satellite Geospatial Intelligence.
Technical workstation layout. No scientific processing here.
"""

from __future__ import annotations

import streamlit as st

from ui.components import metric_card, pipeline_stage, section_header
from ui.status import get_pipeline_status


def render_header() -> None:
    st.markdown(
        """
<div class="sgi-header">
  <div class="sgi-brand">
    <div class="sgi-mark">SGI</div>
    <div>
      <div class="sgi-title">Satellite Geospatial Intelligence</div>
      <div class="sgi-subtitle">Earth Observation / Remote Sensing / Geospatial Analytics</div>
    </div>
  </div>
  <div class="sgi-system"><span class="dot"></span>SYSTEM  ONLINE</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_mission_summary(items, drawn_aoi, latitude, longitude, area_size) -> None:
    if items:
        sorted_items = sorted(
            items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100))
        )
        best_cloud = float(sorted_items[0].properties.get("eo:cloud_cover", 0))
        latest_dates = [item.datetime.date() for item in items if item.datetime]
        latest_date = max(latest_dates) if latest_dates else "—"
    else:
        best_cloud = None
        latest_date = "—"

    cols = st.columns(4)
    with cols[0]:
        metric_card("Scenes", str(len(items)) if items else "0", "catalog results")
    with cols[1]:
        metric_card(
            "Best cloud",
            f"{best_cloud:.2f}%" if best_cloud is not None else "—",
            "lowest cloud cover",
        )
    with cols[2]:
        metric_card("Latest", str(latest_date), "scene date")
    with cols[3]:
        if drawn_aoi:
            metric_card("AOI", "Polygon", "drawn on map")
        else:
            metric_card(
                "AOI",
                f"{latitude:.4f}, {longitude:.4f}",
                f"area {area_size:.2f}°",
            )


def render_geospatial_operations_center(map_panel_func) -> None:
    section_header(
        "Geospatial Operations Center",
        "AOI / Sentinel-2 / Spatial Analysis",
    )
    map_panel_func()


def render_scene_catalog(items, download_callback) -> None:
    section_header("Satellite Archive", "Sentinel-2 catalog results")
    if not items:
        st.caption("Search the Sentinel-2 catalog from Analysis Control to populate the archive.")
        return

    with st.expander(f"Archive · {len(items)} scenes", expanded=False):
        header = st.columns([1.2, 0.8, 0.9, 2.2, 0.9])
        header[0].caption("DATE")
        header[1].caption("CLOUD")
        header[2].caption("QUALITY")
        header[3].caption("SCENE ID")
        header[4].caption("ACTION")

        for idx, item in enumerate(items):
            cloud = float(item.properties.get("eo:cloud_cover", 0))
            date_str = str(item.datetime.date()) if item.datetime else "Unknown"
            if cloud <= 1:
                quality = "Excellent"
            elif cloud <= 5:
                quality = "Good"
            elif cloud <= 10:
                quality = "Acceptable"
            else:
                quality = "Cloudy"

            cols = st.columns([1.2, 0.8, 0.9, 2.2, 0.9])
            cols[0].write(date_str)
            cols[1].markdown(
                f'<span class="sgi-mono">{cloud:.2f}%</span>',
                unsafe_allow_html=True,
            )
            cols[2].write(quality)
            cols[3].markdown(
                f'<span class="sgi-mono">{item.id}</span>',
                unsafe_allow_html=True,
            )
            if cols[4].button("Download", key=f"dl_{idx}_{item.id[:12]}"):
                download_callback(item)


def render_active_scene(data, rgb, false_color) -> None:
    section_header("Active Scene", "Natural color / false color composites")
    if data is None:
        st.caption("Download a scene from the archive to activate analysis.")
        return

    meta = st.columns(4)
    meta[0].markdown(
        f'<div class="sgi-metric"><div class="label">Scene ID</div>'
        f'<div class="value" style="font-size:0.78rem">{data["scene_id"]}</div></div>',
        unsafe_allow_html=True,
    )
    meta[1].markdown(
        f'<div class="sgi-metric"><div class="label">Date</div>'
        f'<div class="value">{data["date"]}</div></div>',
        unsafe_allow_html=True,
    )
    meta[2].markdown(
        f'<div class="sgi-metric"><div class="label">Cloud</div>'
        f'<div class="value">{data["cloud"]:.2f}%</div></div>',
        unsafe_allow_html=True,
    )
    aoi = f'{data.get("latitude", "—")}, {data.get("longitude", "—")}'
    meta[3].markdown(
        f'<div class="sgi-metric"><div class="label">AOI center</div>'
        f'<div class="value" style="font-size:0.85rem">{aoi}</div></div>',
        unsafe_allow_html=True,
    )

    if rgb is not None and false_color is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("NATURAL COLOR")
            st.image(rgb, width="stretch")
        with c2:
            st.caption("FALSE COLOR")
            st.image(false_color, width="stretch")
    elif rgb is not None:
        st.caption("NATURAL COLOR")
        st.image(rgb, width="stretch")
    elif false_color is not None:
        st.caption("FALSE COLOR")
        st.image(false_color, width="stretch")


def _fmt_stat(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def render_spectral_intelligence(ndvi, ndwi, ndbi, index_figure, stats: dict | None = None) -> None:
    section_header("Spectral", "NDVI / NDWI / NDBI")
    if ndvi is None and ndwi is None and ndbi is None:
        st.caption("Spectral indices appear after a scene is processed.")
        return

    stats = stats or {}
    rows = [
        ("NDVI", ndvi, "vegetation", stats.get("ndvi")),
        ("NDWI", ndwi, "water", stats.get("ndwi")),
        ("NDBI", ndbi, "built-up", stats.get("ndbi")),
    ]
    cols = st.columns(3)
    for col, (name, mean_v, hint, full) in zip(cols, rows):
        with col:
            if full:
                metric_card(name, _fmt_stat(full.get("mean", mean_v)), hint)
                st.caption(
                    f"med {_fmt_stat(full.get('median'))} · "
                    f"min {_fmt_stat(full.get('min'))} · "
                    f"max {_fmt_stat(full.get('max'))} · "
                    f"std {_fmt_stat(full.get('std'))}"
                )
            else:
                metric_card(name, _fmt_stat(mean_v), f"mean {hint}")

    if index_figure is not None:
        st.pyplot(index_figure, width="stretch")


def render_land_cover(classification_fig, percentages, area_data) -> None:
    section_header("Land Cover", "Classification from spectral indices")
    if classification_fig is None:
        st.caption("Land-cover map appears after a scene is processed.")
        return

    if percentages:
        keys = list(percentages.keys())
        cols = st.columns(min(len(keys), 5) or 1)
        for i, key in enumerate(keys):
            with cols[i % len(cols)]:
                pct = percentages[key]
                area = None
                if area_data and key in area_data:
                    area = area_data[key]
                hint = f"{area:.3f} km²" if area is not None else "share"
                metric_card(str(key), f"{pct:.1f}%", hint)

    st.pyplot(classification_fig, width="stretch")


def render_change_detection_controls(items):
    """
    Controls for change detection.
    Returns param dict when Analyze is clicked, else None.
    """
    section_header("Change", "Before / After comparison")
    if len(items) < 2:
        st.caption("Search for at least two scenes to enable change detection.")
        return None

    scene_options = {}
    for item in items:
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        label = f"{date_str} · {cloud:.2f}% · {item.id[:12]}"
        scene_options[label] = item
    scene_names = list(scene_options.keys())

    c1, c2 = st.columns(2)
    with c1:
        before_name = st.selectbox(
            "Before",
            scene_names,
            index=len(scene_names) - 1,
            key="change_before",
        )
    with c2:
        after_name = st.selectbox(
            "After",
            scene_names,
            index=0,
            key="change_after",
        )

    c3, c4 = st.columns(2)
    with c3:
        threshold = st.slider(
            "Sensitivity", 0.01, 0.50, 0.10, 0.01, key="change_threshold"
        )
    with c4:
        index_choice = st.selectbox(
            "Index",
            ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built-up"],
            key="change_index",
        )

    if st.button("Analyze changes", type="primary", key="change_detect_btn"):
        return {
            "before_name": before_name,
            "after_name": after_name,
            "threshold": threshold,
            "index_choice": index_choice,
            "scene_options": scene_options,
        }
    return None


def render_change_detection_results() -> None:
    change_result = st.session_state.get("change_result")
    if not change_result:
        return

    stats = change_result["statistics"]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Decrease", f"{stats['decrease_km2']:.3f} km²")
    with c2:
        metric_card("Increase", f"{stats['increase_km2']:.3f} km²")
    with c3:
        metric_card("Total changed", f"{stats['total_changed_km2']:.3f} km²")

    if change_result.get("figure"):
        st.pyplot(change_result["figure"], width="stretch")


def render_geospatial_ai_controls(data, detection_rgb):
    """
    AI inference configuration.
    Returns param dict when Run is clicked, else None.
    """
    section_header("Geospatial AI", "Object detection / inference parameters")
    if data is None or detection_rgb is None:
        st.caption("Download a scene to enable Geospatial AI.")
        return None

    with st.expander("Model configuration", expanded=False):
        st.image(detection_rgb, caption="Input RGB", width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            tile_size = st.selectbox(
                "Tile size", [256, 512, 768, 1024], index=1, key="ai_tile_size"
            )
            overlap = st.slider("Overlap", 0, 256, 64, 16, key="ai_overlap")
        with c2:
            confidence = st.slider(
                "Confidence", 0.10, 0.95, 0.50, 0.05, key="ai_confidence"
            )
            try:
                from src.model_registry import list_models

                model_ids = list_models()
                selected_model = st.selectbox("Model", model_ids, key="ai_model")
            except Exception:
                selected_model = None
            classes = st.multiselect(
                "Classes",
                ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"],
                key="ai_classes",
            )

        if st.button("Run inference", type="primary", key="run_ai_btn"):
            return {
                "model_id": selected_model or "",
                "tile_size": tile_size,
                "overlap": overlap,
                "confidence": confidence,
                "classes": classes,
            }
    return None


def render_geospatial_ai_results(detections, detection_rgb) -> None:
    if not detections:
        return

    try:
        from src.object_detection import detection_summary

        summary = detection_summary(detections)
    except Exception:
        summary = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Detections", str(len(detections)))
    with c2:
        metric_card("Classes", str(len(summary)))
    with c3:
        metric_card("Georef", "Ready" if st.session_state.get("transform") else "—")

    detection_fig = st.session_state.get("detection_figure")
    if detection_fig is not None:
        st.pyplot(detection_fig, width="stretch")

    if st.button("Export GeoJSON", key="export_geojson_btn"):
        st.session_state["export_geojson"] = True


def render_pipeline_status() -> None:
    section_header("Pipeline", "Processing stages")
    status = get_pipeline_status()
    stages = ["Catalog", "Imagery", "Spectral", "Change", "AI"]
    parts = []
    for stage in stages:
        state = status.get(stage, "pending")
        css = {
            "done": "ready",
            "ready": "ready",
            "active": "active",
            "pending": "pending",
            "error": "error",
        }.get(state, "pending")
        label = {
            "done": "READY",
            "ready": "READY",
            "active": "ACTIVE",
            "pending": "PENDING",
            "error": "ERROR",
        }.get(state, state.upper())
        parts.append(
            f'<span class="stage {css}"><span class="dot"></span>'
            f"{stage.upper()}  {label}</span>"
        )
    st.markdown(
        f'<div class="sgi-pipe">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
<div class="sgi-footer">
  Satellite Geospatial Intelligence · Earth Observation · Remote Sensing · Geospatial Analytics<br>
  Spectral values are analytical measurements; interpret with sensor, resolution and preprocessing context.
</div>
        """,
        unsafe_allow_html=True,
    )
