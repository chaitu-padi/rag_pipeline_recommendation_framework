"""
API routes for health checks.
"""
from fastapi import APIRouter, HTTPException
from ...core.composer.pipeline import EnhancedPipelineComposer

router = APIRouter()
composer = EnhancedPipelineComposer()

@router.get("/health")
async def health():
    """Enhanced health check with component status"""
    try:
        kb = composer.knowledge_base
        return {
            "status": "healthy",
            "version": "2.0.0",
            "components": {
                "embedding_models": len(kb.embedding_models),
                "vector_databases": len(kb.vector_databases),
                "chunking_strategies": len(kb.chunking_strategies),
                "llm_models": len(kb.llm_models),
                "domain_expertise": len(kb.domain_expertise),
                "deployment_patterns": len(kb.deployment_patterns)
            },
            "capabilities": [
                "comprehensive_business_analysis",
                "domain_specific_optimization",
                "deployment_flexibility",
                "risk_assessment",
                "implementation_roadmap",
                "future_evolution_planning"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
