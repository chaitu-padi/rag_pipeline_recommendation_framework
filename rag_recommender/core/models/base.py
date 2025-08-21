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

class ContentDomain(Enum):
    GENERAL = "general"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    CUSTOMER_SUPPORT = "customer_support"
    HR_POLICIES = "hr_policies"
    MARKETING = "marketing"
    COMPLIANCE = "compliance"

class VolumeSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"
    MASSIVE = "massive"

class UpdateFrequency(Enum):
    STATIC = "static"
    WEEKLY = "weekly"
    DAILY = "daily"
    HOURLY = "hourly"
    REAL_TIME = "real_time"
    EVENT_DRIVEN = "event_driven"

class UseCase(Enum):
    QUESTION_ANSWERING = "question_answering"
    SEMANTIC_SEARCH = "semantic_search"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    CONVERSATIONAL_AI = "conversational_ai"
    CONTENT_ANALYSIS = "content_analysis"
    COMPLIANCE_CHECKING = "compliance_checking"
    RESEARCH_ASSISTANCE = "research_assistance"
    KNOWLEDGE_DISCOVERY = "knowledge_discovery"

class QueryComplexity(Enum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_STEP = "multi_step"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    INTERPRETIVE = "interpretive"

class ResponseType(Enum):
    BRIEF_ANSWERS = "brief_answers"
    DETAILED_EXPLANATIONS = "detailed_explanations"
    STRUCTURED_SUMMARIES = "structured_summaries"
    CONVERSATIONAL = "conversational"
    CITATION_HEAVY = "citation_heavy"

class AccuracyTolerance(Enum):
    HIGH_TOLERANCE = "high_tolerance"
    MODERATE_TOLERANCE = "moderate_tolerance"
    LOW_TOLERANCE = "low_tolerance"
    ZERO_TOLERANCE = "zero_tolerance"

class LatencyRequirement(Enum):
    REAL_TIME = "real_time"
    INTERACTIVE = "interactive"
    RESPONSIVE = "responsive"
    BATCH_ACCEPTABLE = "batch_acceptable"

class BudgetRange(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNLIMITED = "unlimited"

class TeamExpertise(Enum):
    NON_TECHNICAL = "non_technical"
    BASIC_TECHNICAL = "basic_technical"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

# Core Data Models
@dataclass
class DataCharacteristics:
    """Comprehensive data profiling information"""
    primary_data_type: DataType
    secondary_data_types: List[DataType] = field(default_factory=list)
    document_complexity: DocumentComplexity = DocumentComplexity.MODERATE
    content_domain: ContentDomain = ContentDomain.GENERAL
    current_volume: VolumeSize = VolumeSize.MEDIUM
    average_document_length: str = "medium"
    update_frequency: UpdateFrequency = UpdateFrequency.STATIC
    language_requirements: List[str] = field(default_factory=lambda: ["english"])
    expected_growth_rate: str = "10% monthly"
    metadata_importance: str = "medium"

@dataclass
class UseCaseRequirements:
    """Use case specific requirements"""
    primary_use_case: UseCase
    expected_query_complexity: QueryComplexity
    accuracy_tolerance: AccuracyTolerance
    latency_requirement: LatencyRequirement
    preferred_response_type: ResponseType = ResponseType.DETAILED_EXPLANATIONS
    citation_requirements: str = "helpful"
    expected_queries_per_day: str = "100-1000"
    concurrent_users: str = "1-10"

@dataclass
class BusinessContext:
    """Business related context and constraints"""
    budget_range: BudgetRange
    team_expertise: TeamExpertise
    regulatory_requirements: List[str] = field(default_factory=list)
    data_sensitivity_level: str = "internal"
    maintenance_capability: str = "moderate"
    industry_domain: str = "general"
    integration_requirements: List[str] = field(default_factory=list)

@dataclass
class TechnicalPreferences:
    """Technical implementation preferences"""
    deployment_preference: str = "flexible"
    scalability_requirements: str = "moderate"
    monitoring_depth: str = "standard"
    customization_importance: str = "moderate"

@dataclass
class ComponentSpec:
    """Base specification for pipeline components"""
    name: str
    description: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

@dataclass
class AdvancedChunkingConfig(ComponentSpec):
    """Advanced document chunking configuration"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: str = "semantic"

@dataclass
class AdvancedEmbeddingConfig(ComponentSpec):
    """Advanced embedding model configuration"""
    model_name: str = "default"
    dimension: int = 768
    batch_size: int = 32

@dataclass
class AdvancedVectorDBConfig(ComponentSpec):
    """Advanced vector database configuration"""
    database_type: str = "faiss"
    index_type: str = "flat"
    metric: str = "cosine"

@dataclass
class AdvancedRAGConfig(ComponentSpec):
    """Advanced RAG configuration"""
    retrieval_strategy: str = "hybrid"
    reranking_enabled: bool = True
    result_count: int = 5

@dataclass
class EnhancedIngestionPipeline:
    """Enhanced ingestion pipeline specification"""
    name: str
    description: str
    chunking: AdvancedChunkingConfig
    embedding: AdvancedEmbeddingConfig
    vector_db: AdvancedVectorDBConfig
    when_to_use: str
    trade_offs: List[str] = field(default_factory=list)
    estimated_cost: str = "moderate"

@dataclass
class EnhancedRAGPipeline:
    """Enhanced RAG pipeline specification"""
    name: str
    description: str
    ingestion_pipeline: EnhancedIngestionPipeline
    rag_config: AdvancedRAGConfig
    when_to_use: str
    trade_offs: List[str] = field(default_factory=list)
    estimated_cost: str = "moderate"

@dataclass
class ComprehensiveUserRequirements:
    """Complete user requirements specification"""
    data_characteristics: DataCharacteristics
    use_case_requirements: UseCaseRequirements
    business_context: BusinessContext
    technical_preferences: TechnicalPreferences
    additional_context: str = ""
    # ... (rest of the class implementation)

# ... (rest of the model classes)

@dataclass
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
