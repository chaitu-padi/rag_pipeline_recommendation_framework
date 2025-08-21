"""
Module initialization for API routes.
"""
from .recommendations import router as recommendations_router
from .components import router as components_router
from .health import router as health_router
from .web import router as web_router

__all__ = ['recommendations', 'components', 'health', 'web']
