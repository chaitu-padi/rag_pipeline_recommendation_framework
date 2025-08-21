"""
API route for serving the web interface.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))

@router.get("/", response_class=HTMLResponse)
async def get_form(request):
    """Serve the comprehensive questionnaire form"""
    return templates.TemplateResponse("index.html", {"request": request})
