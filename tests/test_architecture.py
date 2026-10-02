"""Architecture guardrails for the application shell."""
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_application_shell_does_not_duplicate_scene_catalog_markup():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "render_scene_catalog(items" in app
    assert "render_scene_catalog_integrated" not in app
    assert "catalog-table" not in app


def test_scientific_workflows_do_not_depend_on_streamlit():
    for name in ("imagery.py", "change.py"):
        source = (ROOT / "src" / "workflows" / name).read_text(encoding="utf-8")
        assert "import streamlit" not in source
        assert "from streamlit" not in source


def test_ai_is_checkpoint_gated():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "model_available(model_id)" in app
    assert "MODEL UNAVAILABLE" in app


def test_single_application_entrypoint():
    assert (ROOT / "app.py").exists()
    assert (ROOT / "ui" / "mission_control.py").exists()
    assert (ROOT / "ui" / "catalog.py").exists()
    assert (ROOT / "src" / "workflows" / "imagery.py").exists()
    assert (ROOT / "src" / "workflows" / "change.py").exists()
