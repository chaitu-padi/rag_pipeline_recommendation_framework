"""
Enhanced recommendation engine for pipeline components.
"""
from typing import Dict, List, Tuple, Any

from ..models.base import (
    ComprehensiveUserRequirements, AdvancedChunkingConfig,
    ContentDomain, DocumentComplexity, VolumeSize, UseCase,
    QueryComplexity, ResponseType, LatencyRequirement, TeamExpertise
)

class EnhancedRecommendationEngine:
    """Advanced recommendation engine for RAG pipeline components"""
    
    def analyze_comprehensive_requirements(
        self,
        requirements: ComprehensiveUserRequirements
    ) -> Dict[str, float]:
        """Analyze comprehensive user requirements and generate component scores"""
        scores = {}
        
        # Content complexity score
        # use nested dataclasses from ComprehensiveUserRequirements
        dc = requirements.data_characteristics
        uc = requirements.use_case_requirements
        bc = requirements.business_context
        tp = requirements.technical_preferences

        scores["content_complexity"] = self._calculate_content_complexity(
            dc.content_domain,
            dc.document_complexity,
            dc.current_volume
        )
        
        # Usage pattern score
        scores["usage_pattern"] = self._calculate_usage_pattern(
            uc.primary_use_case,
            uc.expected_query_complexity,
            uc.preferred_response_type,
            uc.latency_requirement
        )
        
        # Technical constraints score
        # resource constraints aren't modeled explicitly; derive minimal info
        resource_constraints = {
            "budget_range": bc.budget_range
        }
        scores["technical_constraints"] = self._calculate_technical_constraints(
            bc.team_expertise,
            resource_constraints
        )
        
        return scores

    def recommend_chunking_strategy(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> Tuple[str, AdvancedChunkingConfig]:
        """Recommend optimal chunking strategy and configuration"""
        # Minimal heuristic: if content is complex or documents are long, use semantic chunking
        dc = requirements.data_characteristics
        strategy = "semantic" if dc.document_complexity in (DocumentComplexity.COMPLEX, DocumentComplexity.HIGHLY_COMPLEX) else "basic"
        config = AdvancedChunkingConfig(
            name=f"{strategy.title()} Chunking Strategy",
            description=f"Optimized {strategy} chunking for {dc.document_complexity.value} document complexity",
            chunking_strategy=strategy
        )
        return strategy, config

    def recommend_embedding_models(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> List[str]:
        """Recommend suitable embedding models"""
        # Return a small default list; real logic should consult model catalog
        return ["all-MiniLM-L6-v2"]

    def recommend_vector_databases(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> List[str]:
        """Recommend suitable vector databases"""
        # Minimal default
        return ["faiss"]

    def recommend_llm_models(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> List[str]:
        """Recommend suitable LLM models"""
        # Minimal default LLM suggestions
        return ["gpt-4o-mini"]

    def _calculate_content_complexity(
        self,
        content_domain: ContentDomain,
        document_complexity: DocumentComplexity,
        volume_size: VolumeSize
    ) -> float:
        """Calculate content complexity score"""
        score = 1.0
        # Simple heuristic
        if document_complexity == DocumentComplexity.SIMPLE:
            score = 0.5
        elif document_complexity == DocumentComplexity.MODERATE:
            score = 1.0
        elif document_complexity == DocumentComplexity.COMPLEX:
            score = 1.5
        elif document_complexity == DocumentComplexity.HIGHLY_COMPLEX:
            score = 2.0
        # Volume adjustment
        if volume_size in (VolumeSize.LARGE, VolumeSize.VERY_LARGE, VolumeSize.MASSIVE):
            score *= 1.2
        return score

    def _calculate_usage_pattern(
        self,
        use_case: UseCase,
        query_complexity: QueryComplexity,
        response_type: ResponseType,
        latency_requirement: LatencyRequirement
    ) -> float:
        """Calculate usage pattern score"""
        # Basic scoring heuristic
        score = 1.0
        if query_complexity == QueryComplexity.SIMPLE_FACTUAL:
            score = 0.8
        elif query_complexity == QueryComplexity.MULTI_STEP:
            score = 1.2
        elif query_complexity == QueryComplexity.ANALYTICAL:
            score = 1.5
        # latency preference factor
        if latency_requirement == LatencyRequirement.REAL_TIME:
            score *= 1.3
        return score

    def _calculate_technical_constraints(
        self,
        team_expertise: TeamExpertise,
        resource_constraints: Dict[str, Any]
    ) -> float:
        """Calculate technical constraints score"""
        # Basic heuristic: less expertise -> higher constraint
        score = 1.0
        if team_expertise == TeamExpertise.NON_TECHNICAL:
            score = 1.5
        elif team_expertise == TeamExpertise.BASIC_TECHNICAL:
            score = 1.3
        return score
