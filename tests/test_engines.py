from pathlib import Path

import pytest

from safety.config import load_settings
from safety.engines import EngineUnavailable, make_engine
from safety.jev_client import FakePpeClient, JevPpeClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings():
    s = load_settings(ROOT / "config", env_file=ROOT / "missing.env")
    s.typesafe_api_key = None
    return s


def test_jev_needs_an_api_key(settings):
    with pytest.raises(EngineUnavailable, match="TYPESAFE_API_KEY"):
        make_engine("jev_vlm", settings)
    settings.typesafe_api_key = "test-key"
    engine = make_engine("jev_vlm", settings)
    assert isinstance(engine.checker, JevPpeClient) and engine.min_confidence == settings.jev.min_confidence


def test_fake_jev_works_offline(settings):
    assert isinstance(make_engine("jev_vlm", settings, fake_jev=True).checker, FakePpeClient)


def test_local_engine_without_weights_is_unavailable(settings, tmp_path):
    settings.local_ml.weights = str(tmp_path / "missing.pt")
    with pytest.raises(EngineUnavailable, match="not found"):
        make_engine("local_ml", settings)


def test_unknown_engine_is_rejected(settings):
    with pytest.raises(ValueError):
        make_engine("magic", settings)
