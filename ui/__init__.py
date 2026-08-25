# ui/__init__.py
# Exporta os símbolos principais usados pelo app.py

from .components import metric_card, pipeline_stage
from .status import init_pipeline_status, update_pipeline_status, get_pipeline_status
from .theme import apply_theme

from .layout import (
    render_header,
    render_mission_summary,
    render_geospatial_operations_center,
    render_scene_catalog,
    render_active_scene,
    render_spectral_intelligence,
    render_land_cover,
    render_change_detection_controls,
    render_change_detection_results,
    render_geospatial_ai_controls,
    render_geospatial_ai_results,
    render_pipeline_status,
    render_footer,
)