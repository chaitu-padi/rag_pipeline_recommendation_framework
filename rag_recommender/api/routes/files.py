"""
Static files serving and web routes.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))

# Get the absolute path to the pipeline_configs directory
CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "pipeline_configs"

# Ensure the configs directory exists
CONFIGS_DIR.mkdir(exist_ok=True)

@router.get("/api/v1/download-config/{filename}")
async def download_config(filename: str):
    """Download a generated YAML configuration file."""
    try:
        # Clean the filename and ensure it has .yaml extension
        safe_filename = Path(filename).stem + ".yaml"
        file_path = CONFIGS_DIR / safe_filename
        
        # Verify the file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration file {safe_filename} not found. Please generate recommendations first."
            )
        
        # Verify the file is within the configs directory
        if not str(file_path.resolve()).startswith(str(CONFIGS_DIR.resolve())):
            raise HTTPException(
                status_code=403, 
                detail="Invalid file path"
            )
        
        # Return file as attachment for download
        return FileResponse(
            path=str(file_path),
            filename=safe_filename,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
