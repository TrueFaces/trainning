from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from google.cloud import storage
import io
import functools

from fastapi.security import OAuth2PasswordBearer
from fastapi.logger import logger

from app.config import settings
from tensorflow.keras.models import load_model

ROOT_PATH = "model"


async def upload_file_to_bucket(path: str, file: UploadFile):
    file_path = f'{ROOT_PATH}/{path}/{file.filename}'

    storage_client = storage.Client()
    bucket = storage_client.bucket(settings.bucket)

    blob = bucket.blob(file_path)
    blob.upload_from_file(file.file)

    return blob.public_url


async def download_file_from_bucket(path: str, filename: str):
    file_path = f'{ROOT_PATH}/{path}/{filename}'

    storage_client = storage.Client()
    bucket = storage_client.bucket(settings.bucket)

    blob = bucket.blob(file_path)

    file_stream = io.BytesIO()
    blob.download_to_file(file_stream)
    file_stream.seek(0)

    return StreamingResponse(
        file_stream,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


async def delete_file_from_bucket(path: str):
    file_path = f'{ROOT_PATH}/{path}'

    storage_client = storage.Client()
    bucket = storage_client.bucket(settings.bucket)

    blob = bucket.blob(file_path)
    # Check if file exists
    blob.delete()


@functools.cache
def load_model_from_bucket(path):
    model_path = f'{ROOT_PATH}/{path}'
    # Set up the GCP client
    client = storage.Client()
    bucket = client.get_bucket(settings.bucket)

    # Download the model file from the bucket
    blob = bucket.blob(model_path)
    blob.download_to_filename('model.h5')
    model = load_model('model.h5')

    return model
