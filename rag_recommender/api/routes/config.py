"""
Configuration file download endpoints.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.get("/download-config/{filename}")
async def download_config(filename: str):
    """Download a generated YAML configuration file."""
    try:
        file_path = os.path.join("pipeline_configs", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/x-yaml"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
