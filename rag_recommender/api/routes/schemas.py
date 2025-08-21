"""
API schemas for request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ComprehensiveRequirementsRequest(BaseModel):
    """Comprehensive requirements request schema"""
    # Data characteristics
    primary_data_type: str
    secondary_data_types: List[str] = []
    document_complexity: str
    content_domain: str
    current_volume: str
    average_document_length: str = "medium"
    update_frequency: str = "static"
    language_requirements: List[str] = ["english"]
    expected_growth_rate: str = "10% monthly"
    metadata_importance: str = "medium"

    # Use case requirements
    primary_use_case: str
    expected_query_complexity: str
    accuracy_tolerance: str
    latency_requirement: str
    preferred_response_type: str = "detailed_explanations"
    citation_requirements: str = "helpful"
    expected_queries_per_day: str = "100-1000"
    concurrent_users: str = "1-10"

    # Business context
    budget_range: str
    team_expertise: str
    regulatory_requirements: List[str] = []
    data_sensitivity_level: str
    maintenance_capability: str = "moderate"
    industry_domain: str = "general"
    integration_requirements: List[str] = []

    # Technical preferences
    deployment_preference: str = "flexible"
    scalability_requirements: str = "moderate"
    monitoring_depth: str = "standard"
    customization_importance: str = "moderate"
    additional_context: str = ""

class ComprehensiveRecommendationResponse(BaseModel):
    """Comprehensive recommendation response schema"""
    ingestion_pipelines: List[Dict[str, Any]]
    rag_pipelines: List[Dict[str, Any]]
    requirements_analysis: Dict[str, str]
    trade_off_analysis: Dict[str, str]
    risk_assessment: Dict[str, str]
    implementation_roadmap: List[str]
    success_metrics: List[str]
    monitoring_recommendations: List[str]
    alternative_considerations: List[str]
    future_evolution_path: List[str]
    generated_at: str
