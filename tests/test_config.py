from pathlib import Path

from fer_inference_api.config import Settings


def test_settings_has_model_path():
    settings = Settings()
    assert isinstance(settings.model_path, Path)


def test_settings_has_blazeface_model_path():
    settings = Settings()
    assert isinstance(settings.blazeface_model_path, Path)


def test_resolved_model_path_is_absolute():
    settings = Settings()
    assert settings.resolved_model_path.is_absolute()


def test_resolved_blazeface_path_is_absolute():
    settings = Settings()
    assert settings.resolved_blazeface_path.is_absolute()


def test_model_path_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/custom/path/model.onnx")
    settings = Settings()
    assert settings.model_path == Path("/custom/path/model.onnx")


def test_blazeface_model_path_from_env(monkeypatch):
    monkeypatch.setenv("BLAZEFACE_MODEL_PATH", "/custom/blazeface.onnx")
    settings = Settings()
    assert settings.blazeface_model_path == Path("/custom/blazeface.onnx")


def test_extra_env_ignored(monkeypatch):
    monkeypatch.setenv("UNKNOWN_SETTING", "should_be_ignored")
    settings = Settings()
    assert not hasattr(settings, "UNKNOWN_SETTING")


def test_env_file_model_config(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_PATH=/from/file/model.onnx\n")
    settings = Settings(_env_file=str(env_file))
    assert settings.model_path == Path("/from/file/model.onnx")
