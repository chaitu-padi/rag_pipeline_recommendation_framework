"""
API routes for recommendations.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ...core.models.base import (
    ComprehensiveUserRequirements, DataCharacteristics, UseCaseRequirements,
    BusinessContext, TechnicalPreferences, DataType, DocumentComplexity,
    ContentDomain, VolumeSize, UpdateFrequency, UseCase, QueryComplexity,
    ResponseType, AccuracyTolerance, LatencyRequirement, BudgetRange,
    TeamExpertise
)
from ...core.composer.pipeline import EnhancedPipelineComposer
from .schemas import (
    ComprehensiveRequirementsRequest,
    ComprehensiveRecommendationResponse
)

router = APIRouter()
composer = EnhancedPipelineComposer()

@router.post("/recommend-comprehensive", response_model=ComprehensiveRecommendationResponse)
async def recommend_comprehensive_pipelines(requirements: ComprehensiveRequirementsRequest = None):
    """Generate comprehensive pipeline recommendations based on detailed business requirements"""
    if requirements is None:
        requirements = ComprehensiveRequirementsRequest(
            primary_data_type="text",
            document_complexity="moderate",
            content_domain="general",
            current_volume="medium",
            primary_use_case="question_answering",
            expected_query_complexity="simple_factual",
            accuracy_tolerance="moderate_tolerance",
            latency_requirement="interactive",
            budget_range="moderate",
            team_expertise="intermediate",
            data_sensitivity_level="internal"
        )
    try:
        try:
            # Convert request to internal model
            comprehensive_requirements = ComprehensiveUserRequirements(
                data_characteristics=DataCharacteristics(
                    primary_data_type=DataType(requirements.primary_data_type),
                    secondary_data_types=[DataType(dt) for dt in (requirements.secondary_data_types or [])],
                    document_complexity=DocumentComplexity(requirements.document_complexity),
                    content_domain=ContentDomain(requirements.content_domain),
                    current_volume=VolumeSize(requirements.current_volume),
                    average_document_length=requirements.average_document_length,
                    update_frequency=UpdateFrequency(requirements.update_frequency),
                    language_requirements=requirements.language_requirements,
                    expected_growth_rate=requirements.expected_growth_rate,
                    metadata_importance=requirements.metadata_importance
                ),
                use_case_requirements=UseCaseRequirements(
                    primary_use_case=UseCase(requirements.primary_use_case),
                    expected_query_complexity=QueryComplexity(requirements.expected_query_complexity),
                    accuracy_tolerance=AccuracyTolerance(requirements.accuracy_tolerance),
                    latency_requirement=LatencyRequirement(requirements.latency_requirement),
                    preferred_response_type=ResponseType(requirements.preferred_response_type),
                    citation_requirements=requirements.citation_requirements,
                    expected_queries_per_day=requirements.expected_queries_per_day,
                    concurrent_users=requirements.concurrent_users
                ),
                business_context=BusinessContext(
                    budget_range=BudgetRange(requirements.budget_range),
                    team_expertise=TeamExpertise(requirements.team_expertise),
                    data_sensitivity_level=requirements.data_sensitivity_level,
                    industry_domain=requirements.industry_domain
                ),
                technical_preferences=TechnicalPreferences(
                    scalability_requirements=requirements.scalability_requirements,
                    monitoring_depth=requirements.monitoring_depth
                ),
                additional_context=requirements.additional_context
            )
        except AttributeError as e:
            # Handle case where a required field is missing
            missing_field = str(e).split("'")[3] if "has no attribute" in str(e) else "unknown field"
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required field: {missing_field}"
            )

        # Generate recommendations
        result = composer.generate_comprehensive_recommendations(comprehensive_requirements)

        # Convert to response format
        return ComprehensiveRecommendationResponse(
            ingestion_pipelines=[
                pipeline.__dict__ for pipeline in result.ingestion_pipelines
            ],
            rag_pipelines=[
                pipeline.__dict__ for pipeline in result.rag_pipelines
            ],
            requirements_analysis=result.requirements_analysis,
            trade_off_analysis=result.trade_off_analysis,
            risk_assessment=result.risk_assessment,
            implementation_roadmap=result.implementation_roadmap,
            success_metrics=result.success_metrics,
            monitoring_recommendations=result.monitoring_recommendations,
            alternative_considerations=result.alternative_considerations,
            future_evolution_path=result.future_evolution_path,
            generated_at=result.generated_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
