"""
API route for serving the web interface.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))

@router.get("/")
async def root():
    """Redirect to form"""
    return {"message": "Welcome to RAG Pipeline Recommender! Use /form for the questionnaire."}

@router.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    """Serve the comprehensive questionnaire form"""
    return templates.TemplateResponse("index.html", context={"request": request})
