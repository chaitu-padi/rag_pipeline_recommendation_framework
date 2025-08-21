"""
API routes for component information.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ...core.composer.pipeline import EnhancedPipelineComposer

router = APIRouter()
composer = EnhancedPipelineComposer()

@router.get("/components-comprehensive")
async def list_comprehensive_components():
    """List all available components with comprehensive specifications"""
    try:
        kb = composer.knowledge_base

        return {
            "embedding_models": {
                name: {
                    "provider": spec.provider,
                    "dimensions": spec.config.get("dimensions", 0),
                    "cost_tier": spec.cost_tier,
                    "domain_suitability": spec.domain_suitability,
                    "language_support": spec.language_support,
                    "deployment_options": spec.deployment_options,
                    "capabilities": spec.capabilities,
                    "limitations": spec.limitations
                }
                for name, spec in kb.embedding_models.items()
            },
            "vector_databases": {
                name: {
                    "provider": spec.provider,
                    "cost_tier": spec.cost_tier,
                    "scalability_rating": spec.scalability_rating,
                    "deployment_options": spec.deployment_options,
                    "capabilities": spec.capabilities,
                    "limitations": spec.limitations
                }
                for name, spec in kb.vector_databases.items()
            },
            "chunking_strategies": {
                name: {
                    "description": strategy["description"],
                    "complexity": strategy["complexity"],
                    "computational_cost": strategy["computational_cost"],
                    "domain_suitability": strategy["domain_suitability"],
                    "best_for": strategy["best_for"],
                    "limitations": strategy["limitations"]
                }
                for name, strategy in kb.chunking_strategies.items()
            },
            "llm_models": {
                name: {
                    "provider": spec.provider,
                    "cost_tier": spec.cost_tier,
                    "context_window": spec.config.get("context_window", 0),
                    "domain_suitability": spec.domain_suitability,
                    "deployment_options": spec.deployment_options,
                    "capabilities": spec.capabilities,
                    "limitations": spec.limitations
                }
                for name, spec in kb.llm_models.items()
            },
            "domain_expertise": list(kb.domain_expertise.keys()),
            "deployment_patterns": list(kb.deployment_patterns.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/domain-recommendations/{domain}")
async def get_domain_recommendations(domain: str):
    """Get domain-specific recommendations and best practices"""
    try:
        recommendations = composer.knowledge_base.get_domain_recommendations(domain)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

        return {
            "domain": domain,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
