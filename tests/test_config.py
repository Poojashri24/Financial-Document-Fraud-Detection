from backend.core.config import settings


def test_settings_loads_basic_values():
    assert settings.app_name == "financial-doc-fraud-detection"
    assert settings.environment in {"dev", "prod", "test"}
