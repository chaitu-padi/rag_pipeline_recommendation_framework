"""
Core data models for the RAG Pipeline Recommendation Framework.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime

__all__ = [
    'DataType', 'DocumentComplexity', 'ContentDomain', 'VolumeSize', 
    'UpdateFrequency', 'UseCase', 'QueryComplexity', 'ResponseType',
    'AccuracyTolerance', 'LatencyRequirement', 'BudgetRange', 'TeamExpertise',
    'DataCharacteristics', 'UseCaseRequirements', 'BusinessContext',
    'TechnicalPreferences', 'ComprehensiveUserRequirements', 'ComponentSpec',
    'AdvancedChunkingConfig', 'AdvancedEmbeddingConfig', 'AdvancedVectorDBConfig',
    'AdvancedRAGConfig', 'EnhancedIngestionPipeline', 'EnhancedRAGPipeline',
    'ComprehensiveRecommendationResult'
]

# Core Enums
class DataType(Enum):
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    XML = "xml"
    EMAIL = "email"
    CODE = "code"
    MIXED = "mixed"

class DocumentComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

# ... (rest of the enum classes)

# Core Data Models
@dataclass
class DataCharacteristics:
    """Comprehensive data profiling information"""
    primary_data_type: DataType
    secondary_data_types: List[DataType] = field(default_factory=list)
    document_complexity: DocumentComplexity = DocumentComplexity.MODERATE
    content_domain: ContentDomain = ContentDomain.GENERAL
    # ... (rest of the class implementation)

# ... (rest of the model classes)

@dataclass
class ComprehensiveRecommendationResult:
    """Enhanced recommendation result with detailed analysis"""
    user_requirements: ComprehensiveUserRequirements
    ingestion_pipelines: List[EnhancedIngestionPipeline]
    rag_pipelines: List[EnhancedRAGPipeline]
    requirements_analysis: Dict[str, str] = field(default_factory=dict)
    trade_off_analysis: Dict[str, str] = field(default_factory=dict)
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    implementation_roadmap: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    alternative_considerations: List[str] = field(default_factory=list)
    future_evolution_path: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
