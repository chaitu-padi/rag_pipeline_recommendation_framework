"""
Main FastAPI application for the RAG Pipeline Recommendation Framework.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from .routes import recommendations, components, health, web
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Enhanced RAG Pipeline Recommender",
    description="Comprehensive intelligent recommendations for RAG and embedding ingestion pipelines based on detailed business analysis",
    version="2.0.0"
)

# Mount static files
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent.parent / "templates" / "static")), name="static")

# Include routers
app.include_router(web.router, prefix="")  # Root path
app.include_router(recommendations.router, prefix="/api/v1")
app.include_router(components.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

# Import and include files router for downloads
from .routes import files
app.include_router(files.router, prefix="")

# Add error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={
        "error": "Internal server error",
        "detail": str(exc),
        "path": request.url.path
    })

@app.exception_handler(404)
async def not_found_error_handler(request, exc):
    return JSONResponse(status_code=404, content={
        "error": "Not found",
        "detail": str(exc),
        "path": request.url.path
    })

def run_server():
    """Start the FastAPI server"""
    print("🚀 Starting Enhanced RAG Pipeline Recommender v2.0...")
    print("📋 Comprehensive questionnaire: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    print("📊 Component catalog: http://localhost:8000/components-comprehensive")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    run_server()
