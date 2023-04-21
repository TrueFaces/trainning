from fastapi import APIRouter, HTTPException, Depends, UploadFile, Request
from app.utils.storage import upload_file_to_bucket, delete_file_from_bucket
from pydantic import BaseModel
import time

router = APIRouter(prefix="/train", tags=["train"])


@router.post("/")
def start_train_model():
    # Call Preprocess

    # Read images

    # ...

    # Train model
    return


@router.post("/upload")
async def upload_model(file: UploadFile):
    file.filename = f"{int(time.time())}_{file.filename}"
    
    path = await upload_file_to_bucket("test_models", file)

    #Check errors

    return {"status": "ok"}

@router.delete("/remove")
async def remove_model(filename):
    
    await delete_file_from_bucket(filename=filename)

    #Check errors
    
    return {"status": "ok"}
