import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from src.config import settings
from src.database.connection import init_db
from src.api.routes import router
from src.utils.logger import logger

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Multi-Modal AI System for Brain Tumor Classification, Explainability, and Automated Radiology Reporting.",
    version="1.0.0"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded MRI images and heatmaps as static files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.HEATMAP_DIR, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/static/heatmaps", StaticFiles(directory=settings.HEATMAP_DIR), name="heatmaps")

# Register routes
app.include_router(router)


@app.on_event("startup")
def on_startup():
    logger.info("Starting up FastAPI server...")
    init_db()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)