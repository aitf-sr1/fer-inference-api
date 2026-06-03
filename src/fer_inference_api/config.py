from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: Path = Path("./models")
    blazeface_model_path: Path = Path("./assets/blaze_face_short_range.onnx")
    max_cached_models: int = 2
    log_level: str = "info"

    @property
    def resolved_models_dir(self) -> Path:
        return self.models_dir.resolve()

    @property
    def resolved_blazeface_path(self) -> Path:
        return self.blazeface_model_path.resolve()


settings = Settings()
