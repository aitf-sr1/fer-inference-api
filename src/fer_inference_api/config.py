from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Path("./models/convnextv2_femto_v3(crop).onnx")
    blazeface_model_path: Path = Path("./assets/blaze_face_short_range.onnx")
    mock_mode: bool = False
    onnx_intra_threads: int = 0
    onnx_inter_threads: int = 0
    gpu_mem_limit: int = 1_073_741_824

    @property
    def resolved_model_path(self) -> Path:
        return self.model_path.resolve()

    @property
    def resolved_blazeface_path(self) -> Path:
        return self.blazeface_model_path.resolve()


settings = Settings()
