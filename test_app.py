import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_app_runs_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    # pick the synthetic demo and run a local-only (no Gemini) search
    at.radio[0].set_value("Synthetic demo").run()
    assert not at.exception
    assert "df" in at.session_state
    assert any(b.label.endswith("Run search") for b in at.button)
    next(b for b in at.button if b.label.endswith("Run search")).click().run()
    assert not at.exception, at.exception
    assert at.session_state["found"], "search produced no candidates"
