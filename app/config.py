from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    app_name: str = "Truefaces Trainning"
    bucket: str = os.getenv("BUCKET")

settings = Settings()