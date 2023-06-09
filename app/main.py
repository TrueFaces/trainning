from fastapi import Depends, FastAPI

## Add Doc Security
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from fastapi.logger import logger
import logging

from app.routes import train

app = FastAPI()

# Logs
uvicorn_logger = logging.getLogger('uvicorn.error')
logger.handlers = uvicorn_logger.handlers

# Dependency
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# app.include_router(auth.router)
app.include_router(train.router)

@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation():
    return get_redoc_html(openapi_url="/openapi.json", title="docs")


@app.get("/")
async def root():
    return {"status": "running"}