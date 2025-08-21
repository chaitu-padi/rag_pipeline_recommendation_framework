"""
Main FastAPI application for the RAG Pipeline Recommendation Framework.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
from .routes import recommendations, components, health, web

app = FastAPI(
    title="Enhanced RAG Pipeline Recommender",
    description="Comprehensive intelligent recommendations for RAG and embedding ingestion pipelines based on detailed business analysis",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Include routers
app.include_router(web.router)  # Include this first to handle root path
app.include_router(recommendations.router)
app.include_router(components.router)
app.include_router(health.router)

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
