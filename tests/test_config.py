import os
from pathlib import Path

from fer_inference_api.config import Settings


def test_default_model_path():
    settings = Settings()
    assert settings.model_path == Path("./models/model.onnx")


def test_default_blazeface_model_path():
    settings = Settings()
    assert settings.blazeface_model_path == Path("./assets/blaze_face_short_range.onnx")


def test_default_log_level():
    settings = Settings()
    assert settings.log_level == "info"


def test_resolved_model_path():
    settings = Settings()
    assert settings.resolved_model_path == Path("./models/model.onnx").resolve()


def test_resolved_blazeface_path():
    settings = Settings()
    assert settings.resolved_blazeface_path == Path(
        "./assets/blaze_face_short_range.onnx"
    ).resolve()


def test_model_path_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/custom/path/model.onnx")
    settings = Settings()
    assert settings.model_path == Path("/custom/path/model.onnx")


def test_blazeface_model_path_from_env(monkeypatch):
    monkeypatch.setenv("BLAZEFACE_MODEL_PATH", "/custom/blazeface.onnx")
    settings = Settings()
    assert settings.blazeface_model_path == Path("/custom/blazeface.onnx")


def test_log_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.log_level == "debug"


def test_extra_env_ignored(monkeypatch):
    monkeypatch.setenv("UNKNOWN_SETTING", "should_be_ignored")
    settings = Settings()
    assert not hasattr(settings, "UNKNOWN_SETTING")


def test_env_file_model_config(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_PATH=/from/file/model.onnx\nLOG_LEVEL=warning\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    # Settings loads .env from cwd, so we override via model_config
    settings = Settings(_env_file=str(env_file))
    assert settings.model_path == Path("/from/file/model.onnx")
    assert settings.log_level == "warning"
