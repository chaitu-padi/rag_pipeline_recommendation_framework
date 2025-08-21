"""
Enhanced pipeline composer for generating comprehensive recommendations.
"""
from typing import List, Tuple, Dict, Any
from ..models.base import (
    ComprehensiveUserRequirements, ComprehensiveRecommendationResult,
    EnhancedIngestionPipeline, EnhancedRAGPipeline, AdvancedChunkingConfig,
    AdvancedEmbeddingConfig, AdvancedVectorDBConfig, AdvancedRAGConfig,
    ContentDomain, DocumentComplexity, VolumeSize, UseCase, QueryComplexity,
    ResponseType, LatencyRequirement, TeamExpertise, BudgetRange,
    AccuracyTolerance
)
from ..knowledge.base import EnhancedKnowledgeBase
from ..engine.recommendation import EnhancedRecommendationEngine

class EnhancedPipelineComposer:
    """Advanced pipeline composer for comprehensive recommendations"""

    def __init__(self):
        """Initialize pipeline composer with knowledge base and recommendation engine"""
        import logging
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = EnhancedKnowledgeBase()
        self.engine = EnhancedRecommendationEngine()
        self.logger.info("Initialized EnhancedPipelineComposer with knowledge base and recommendation engine")

    def generate_comprehensive_recommendations(
        self, 
        requirements: ComprehensiveUserRequirements
    ) -> ComprehensiveRecommendationResult:
        """Generate complete pipeline recommendations with comprehensive analysis"""

        # Comprehensive requirement analysis
        scores = self.engine.analyze_comprehensive_requirements(requirements)

        # Get intelligent component recommendations
        chunking_strategy, chunking_config = self.engine.recommend_chunking_strategy(requirements, scores)
        embedding_models = self.engine.recommend_embedding_models(requirements, scores)
        vector_dbs = self.engine.recommend_vector_databases(requirements, scores)
        llm_models = self.engine.recommend_llm_models(requirements, scores)

        # Generate enhanced ingestion pipelines
        ingestion_pipelines = self._create_enhanced_ingestion_pipelines(
            requirements, scores, chunking_strategy, chunking_config, 
            embedding_models, vector_dbs
        )

        # Generate enhanced RAG pipelines
        rag_pipelines = self._create_enhanced_rag_pipelines(
            requirements, scores, llm_models
        )

        # Generate comprehensive analysis and guidance
        result = ComprehensiveRecommendationResult(
            user_requirements=requirements,
            ingestion_pipelines=ingestion_pipelines,
            rag_pipelines=rag_pipelines,
            requirements_analysis=self._analyze_requirements_comprehensively(requirements, scores),
            trade_off_analysis=self._analyze_trade_offs(requirements, scores),
            risk_assessment=self._assess_risks(requirements, scores),
            implementation_roadmap=self._create_implementation_roadmap(requirements, scores),
            success_metrics=self._define_success_metrics(requirements),
            monitoring_recommendations=self._create_monitoring_recommendations(requirements),
            alternative_considerations=self._generate_alternative_considerations(requirements, scores),
            future_evolution_path=self._create_evolution_path(requirements)
        )

        # Generate YAML configurations for each pipeline
        import os
        from .yaml_generator import generate_yaml_config, save_yaml_config
        
        os.makedirs("pipeline_configs", exist_ok=True)
        
        # Generate detailed configs with benchmarks
        for i, pipeline in enumerate(result.rag_pipelines):
            config = generate_yaml_config(pipeline, 
                                       data_type=requirements.data_characteristics.primary_data_type.value)
            
            # Save config to file
            config_path = f"pipeline_configs/{pipeline.name.lower().replace(' ', '_')}_config.yaml"
            save_yaml_config(config, config_path)
            
            # Add config path to pipeline metadata
            pipeline.configuration_file = config_path
            
            # Enhance analysis with benchmark details
            pipeline.performance_analysis = self._analyze_performance_metrics(pipeline, config)
            pipeline.resource_requirements = self._analyze_resource_requirements(pipeline, config)
            pipeline.cost_analysis = self._analyze_cost_metrics(pipeline, config)
            pipeline.scaling_characteristics = self._analyze_scaling_characteristics(pipeline, config)
            
        return result

    def _create_enhanced_ingestion_pipelines(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float],
        chunking_strategy: str,
        chunking_config: AdvancedChunkingConfig,
        embedding_models: List[str],
        vector_dbs: List[str]
    ) -> List[EnhancedIngestionPipeline]:
        """Create enhanced ingestion pipelines based on recommendations"""
        pipelines = []

        # High Performance Pipeline
        pipelines.append(
            EnhancedIngestionPipeline(
                name="High Performance Pipeline",
                description="Optimized for maximum performance and accuracy",
                chunking=AdvancedChunkingConfig(
                    name="Advanced Semantic Chunking",
                    description="Semantic-aware document chunking with optimal parameters",
                    chunking_strategy="sentence",
                    configuration={
                        "chunk_size": 1000,
                        "overlap": 100,
                        "combine_small_chunks": True,
                        "min_chunk_size": 100,
                        "sentence_options": {
                            "language": "english",
                            "respect_breaks": True
                        },
                        "smart_splitting": True
                    }
                ),
                embedding=AdvancedEmbeddingConfig(
                    name="High-Performance Embeddings",
                    description="State-of-the-art embedding model with GPU optimization",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    configuration={
                        "batch_size": 256,
                        "normalize": True,
                        "dimension": 384,
                        "optimization": {
                            "use_gpu": True,
                            "half_precision": True,
                            "num_workers": 8
                        }
                    }
                ),
                vector_db=AdvancedVectorDBConfig(
                    name="Qdrant Vector Store",
                    description="High-performance vector database with advanced indexing",
                    database_type="qdrant",
                    configuration={
                        "vector_index": {
                            "type": "hnsw",
                            "params": {
                                "m": 16,
                                "ef_construct": 256,
                                "ef_runtime": 128,
                                "full_scan_threshold": 10000
                            }
                        },
                        "quantization": {
                            "enabled": True,
                            "always_ram": True,
                            "quantum": 0.01
                        },
                        "batch_size": 300,
                        "timeout": 300
                    }
                ),
                when_to_use="For production environments requiring maximum performance",
                trade_offs=["Highest performance", "Higher resource requirements", "GPU recommended"]
            )
        )

        # Balanced Pipeline
        pipelines.append(
            EnhancedIngestionPipeline(
                name="Balanced Pipeline",
                description="Optimal balance between performance and resource usage",
                chunking=AdvancedChunkingConfig(
                    name="Hybrid Chunking",
                    description="Combined fixed-length and semantic chunking",
                    chunking_strategy="sliding_window",
                    configuration={
                        "chunk_size": 500,
                        "overlap": 50,
                        "combine_small_chunks": True,
                        "min_chunk_size": 100,
                        "normalize_whitespace": True
                    }
                ),
                embedding=AdvancedEmbeddingConfig(
                    name="Balanced Embeddings",
                    description="Efficient embedding model for general use",
                    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                    configuration={
                        "batch_size": 128,
                        "normalize": True,
                        "dimension": 768,
                        "optimization": {
                            "use_gpu": False,
                            "num_workers": 4
                        }
                    }
                ),
                vector_db=AdvancedVectorDBConfig(
                    name="Milvus Vector Store",
                    description="Scalable vector database with good performance",
                    database_type="milvus",
                    configuration={
                        "metric_type": "IP",
                        "index_type": "IVF_SQ8",
                        "params": {
                            "nlist": 1024,
                            "nprobe": 16
                        },
                        "batch_size": 200
                    }
                ),
                when_to_use="For most production use cases with balanced requirements",
                trade_offs=["Good performance", "Moderate resource usage", "CPU-friendly"]
            )
        )

        # Resource-Efficient Pipeline
        pipelines.append(
            EnhancedIngestionPipeline(
                name="Resource-Efficient Pipeline",
                description="Optimized for minimal resource consumption",
                chunking=AdvancedChunkingConfig(
                    name="Simple Chunking",
                    description="Lightweight document chunking strategy",
                    chunking_strategy="fixed_length",
                    configuration={
                        "chunk_size": 256,
                        "overlap": 20,
                        "combine_small_chunks": False,
                        "normalize_whitespace": True,
                        "preserve_line_breaks": True
                    }
                ),
                embedding=AdvancedEmbeddingConfig(
                    name="Lightweight Embeddings",
                    description="Resource-efficient embedding model",
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    configuration={
                        "batch_size": 64,
                        "normalize": True,
                        "dimension": 384,
                        "optimization": {
                            "use_gpu": False,
                            "num_workers": 2
                        }
                    }
                ),
                vector_db=AdvancedVectorDBConfig(
                    name="FAISS Vector Store",
                    description="Efficient vector database with minimal overhead",
                    database_type="faiss",
                    configuration={
                        "metric_type": "COSINE",
                        "index_type": "IndexFlatL2",
                        "batch_size": 100
                    }
                ),
                when_to_use="For development, testing, or resource-constrained environments",
                trade_offs=["Minimal resource usage", "Suitable for development", "Good enough performance"]
            )
        )
        return pipelines

    def _calculate_performance_weight(self, use_case_reqs) -> float:
        """Calculate performance weight based on use case requirements"""
        weight = 0.0
        if use_case_reqs.latency_requirement == LatencyRequirement.REAL_TIME:
            weight += 1.0
        elif use_case_reqs.latency_requirement == LatencyRequirement.INTERACTIVE:
            weight += 0.75
        
        # Add weight for high query volume
        if getattr(use_case_reqs, 'expected_queries_per_day', '1-100') in ['1000-10000', '10000+']:
            weight += 0.5
            
        return weight

    def _calculate_cost_weight(self, budget_range: BudgetRange) -> float:
        """Calculate cost weight based on budget constraints"""
        budget_weights = {
            BudgetRange.MINIMAL: 1.0,
            BudgetRange.LOW: 0.8,
            BudgetRange.MODERATE: 0.5,
            BudgetRange.HIGH: 0.2,
            BudgetRange.UNLIMITED: 0.0
        }
        return budget_weights.get(budget_range, 0.5)

    def _calculate_accuracy_weight(self, accuracy_tolerance: AccuracyTolerance, content_domain: ContentDomain) -> float:
        """Calculate accuracy weight based on tolerance and domain"""
        weight = 0.0
        
        # Base weight from accuracy tolerance
        tolerance_weights = {
            AccuracyTolerance.ZERO_TOLERANCE: 1.0,
            AccuracyTolerance.LOW_TOLERANCE: 0.8,
            AccuracyTolerance.MODERATE_TOLERANCE: 0.5,
            AccuracyTolerance.HIGH_TOLERANCE: 0.2
        }
        weight += tolerance_weights.get(accuracy_tolerance, 0.5)
        
        # Additional weight for critical domains
        critical_domains = {
            ContentDomain.LEGAL: 0.3,
            ContentDomain.MEDICAL: 0.3,
            ContentDomain.FINANCIAL: 0.3,
            ContentDomain.COMPLIANCE: 0.3,
            ContentDomain.TECHNICAL: 0.2
        }
        weight += critical_domains.get(content_domain, 0.0)
        
        return min(weight, 1.0)  # Cap at 1.0

    def _calculate_memory_weight(self, scalability_req: str, volume_size: VolumeSize) -> float:
        """Calculate memory weight based on scalability and volume"""
        weight = 0.0
        
        # Base weight from scalability requirements
        if scalability_req == 'minimal':
            weight += 0.2
        elif scalability_req == 'moderate':
            weight += 0.5
        elif scalability_req in ['high', 'extreme']:
            weight += 0.8
            
        # Additional weight from data volume
        volume_weights = {
            VolumeSize.TINY: 0.0,
            VolumeSize.SMALL: 0.1,
            VolumeSize.MEDIUM: 0.2,
            VolumeSize.LARGE: 0.3,
            VolumeSize.VERY_LARGE: 0.4,
            VolumeSize.MASSIVE: 0.5
        }
        weight += volume_weights.get(volume_size, 0.2)
        
        return min(weight, 1.0)  # Cap at 1.0

    def _create_enhanced_rag_pipelines(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float],
        llm_models: List[str]
    ) -> List[EnhancedRAGPipeline]:
        """Create enhanced RAG pipelines based on recommendations using benchmark data"""
        from ..benchmarks.model_benchmarks import (
            EMBEDDING_MODELS_BENCHMARKS,
            VECTOR_DB_BENCHMARKS,
            CHUNKING_BENCHMARKS,
            BenchmarkMetrics
        )
        from ..models.base import (
            BudgetRange,
            AccuracyTolerance,
            LatencyRequirement,
            ContentDomain,
            VolumeSize
        )
        
        pipelines = []
        default_model = llm_models[0] if llm_models else "gpt-4-turbo"
        
        # Calculate weights based on simplified requirements and scores
        weights = {
            'performance': (
                scores.get('performance', 0.0) * 2 + 
                self._calculate_performance_weight(requirements.use_case_requirements)
            ),
            'cost': (
                scores.get('cost_efficiency', 0.0) * 2 + 
                self._calculate_cost_weight(requirements.business_context.budget_range)
            ),
            'accuracy': (
                scores.get('accuracy', 0.0) * 2 + 
                self._calculate_accuracy_weight(
                    requirements.use_case_requirements.accuracy_tolerance,
                    requirements.data_characteristics.content_domain
                )
            ),
            'memory': (
                self._calculate_memory_weight(
                    requirements.technical_preferences.scalability_requirements,
                    requirements.data_characteristics.current_volume
                )
            )
        }
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        def score_component(metrics: BenchmarkMetrics) -> float:
            return (
                weights['performance'] * (1/metrics.latency_ms + metrics.throughput/100000)/2 +
                weights['cost'] * (1 - metrics.cost_per_1k/0.001) +
                weights['accuracy'] * metrics.accuracy +
                weights['memory'] * (1 - metrics.memory_mb/8192)
            )
        
        # Score and select components based on requirements
        # Get required data types
        data_types = [requirements.data_characteristics.primary_data_type.value] + [dt.value for dt in requirements.data_characteristics.secondary_data_types]
        
        # Score all embedding models - they should generally work with any text
        embedding_scores = {
            name: score_component(bench.metrics)
            for name, bench in EMBEDDING_MODELS_BENCHMARKS.items()
            if any(fmt in bench.supported_formats for fmt in ["text", "documents", *data_types])
        }
        
        # Vector DBs should work with any embeddings
        vector_db_scores = {
            name: score_component(bench.metrics)
            for name, bench in VECTOR_DB_BENCHMARKS.items()
            if "dense vectors" in bench.supported_formats  # All our vector DBs should support this
        }
        
        # Check chunking strategies that support our data types
        chunking_scores = {
            name: score_component(bench.metrics)
            for name, bench in CHUNKING_BENCHMARKS.items()
            if (
                # Either supports one of our data types
                any(fmt in bench.supported_formats for fmt in data_types)
                # Or supports general text/documents
                or any(fmt in bench.supported_formats for fmt in ["text", "simple_documents"])
            )
        }
        
        # If no specific chunking strategy found, fall back to semantic chunking
        if not chunking_scores:
            chunking_scores = {
                "semantic": score_component(CHUNKING_BENCHMARKS["semantic"].metrics)
            }
        
        # Get top components for each profile
        def get_top_n(scores: Dict[str, float], n: int) -> List[str]:
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Enterprise Profile
        enterprise_components = {
            'embedding': get_top_n(embedding_scores, 1)[0][0],
            'vector_db': get_top_n(vector_db_scores, 1)[0][0],
            'chunking': get_top_n(chunking_scores, 1)[0][0]
        }
        
        # Balanced Profile
        balanced_weights = {k: 0.33 for k in weights}
        balanced_scores = {
            'embedding': {name: sum(v * balanced_weights[k] for k, v in weights.items()) 
                        for name in embedding_scores},
            'vector_db': {name: sum(v * balanced_weights[k] for k, v in weights.items())
                         for name in vector_db_scores},
            'chunking': {name: sum(v * balanced_weights[k] for k, v in weights.items())
                        for name in chunking_scores}
        }
        balanced_components = {
            'embedding': get_top_n(balanced_scores['embedding'], 1)[0][0],
            'vector_db': get_top_n(balanced_scores['vector_db'], 1)[0][0],
            'chunking': get_top_n(balanced_scores['chunking'], 1)[0][0]
        }
        
        # Resource-Efficient Profile
        resource_weights = {'cost': 0.4, 'memory': 0.4, 'performance': 0.1, 'accuracy': 0.1}
        resource_scores = {
            'embedding': {name: sum(v * resource_weights[k] for k, v in weights.items())
                        for name in embedding_scores},
            'vector_db': {name: sum(v * resource_weights[k] for k, v in weights.items())
                         for name in vector_db_scores},
            'chunking': {name: sum(v * resource_weights[k] for k, v in weights.items())
                        for name in chunking_scores}
        }
        resource_components = {
            'embedding': get_top_n(resource_scores['embedding'], 1)[0][0],
            'vector_db': get_top_n(resource_scores['vector_db'], 1)[0][0],
            'chunking': get_top_n(resource_scores['chunking'], 1)[0][0]
        }
        
        # Log selection reasoning
        self.logger.info("Component Selection Analysis:")
        profiles = {
            'Enterprise': enterprise_components,
            'Balanced': balanced_components,
            'Resource-Efficient': resource_components
        }
        
        # Create ingestion pipelines with benchmark-optimized components
        enterprise_pipeline = self._create_enhanced_ingestion_pipelines(
            requirements, 
            scores,
            enterprise_components['chunking'],
            CHUNKING_BENCHMARKS[enterprise_components['chunking']],
            [enterprise_components['embedding']],
            [enterprise_components['vector_db']]
        )[0]
        
        balanced_pipeline = self._create_enhanced_ingestion_pipelines(
            requirements,
            balanced_scores,
            balanced_components['chunking'],
            CHUNKING_BENCHMARKS[balanced_components['chunking']],
            [balanced_components['embedding']],
            [balanced_components['vector_db']]
        )[0]
        
        resource_pipeline = self._create_enhanced_ingestion_pipelines(
            requirements,
            resource_scores,
            resource_components['chunking'],
            CHUNKING_BENCHMARKS[resource_components['chunking']],
            [resource_components['embedding']],
            [resource_components['vector_db']]
        )[0]
        

        # High-Performance RAG Pipeline
        # Create the Enterprise Pipeline with benchmark-optimized settings
        enterprise_config = {
            "retrieval": {
                "enabled": True,
                "top_k": int(10 * (VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.accuracy)),
                "score_threshold": max(0.3, EMBEDDING_MODELS_BENCHMARKS[enterprise_components['embedding']].metrics.accuracy - 0.1),
                "metric_type": "COSINE",
                "nprobe": min(512, int(VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.throughput / 100)),
                "rerank_results": True,
                "diversity_weight": 0.2
            },
            "llm": {
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 2000,
                "context_window": 8192
            },
            "hybrid_search": {
                "enabled": True,
                "keyword_weight": 0.3,
                "semantic_weight": 0.7
            },
            "performance": {
                "batch_size": int(VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.throughput / 1000),
                "max_concurrent": 8,
                "cache_size_mb": int(VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.memory_mb / 4)
            }
        }
        
        pipelines.append(
            EnhancedRAGPipeline(
                name="Enterprise RAG Pipeline",
                description=f"High-performance RAG pipeline using {enterprise_components['embedding']} embeddings and {enterprise_components['vector_db']} vector store",
                ingestion_pipeline=enterprise_pipeline,
                rag_config=AdvancedRAGConfig(
                    name="Advanced RAG",
                    description=f"Enterprise-grade RAG optimized for {requirements.data_characteristics.primary_data_type.value} and {', '.join(dt.value for dt in requirements.data_characteristics.secondary_data_types)}",
                    configuration=enterprise_config
                ),
                when_to_use=f"Optimal for {EMBEDDING_MODELS_BENCHMARKS[enterprise_components['embedding']].best_use_cases[0]}",
                trade_offs=[
                    f"Best quality ({EMBEDDING_MODELS_BENCHMARKS[enterprise_components['embedding']].metrics.accuracy:.2f} accuracy)",
                    f"High throughput ({VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.throughput} vectors/sec)",
                    f"Memory usage: {VECTOR_DB_BENCHMARKS[enterprise_components['vector_db']].metrics.memory_mb}MB",
                    f"Estimated cost: ${EMBEDDING_MODELS_BENCHMARKS[enterprise_components['embedding']].metrics.cost_per_1k:.4f}/1K tokens"
                ]
            )
        )

        # Balanced RAG Pipeline
        # Create the Balanced Pipeline with benchmark-based settings
        balanced_config = {
            "retrieval": {
                "enabled": True,
                "top_k": int(5 * (VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.accuracy)),
                "score_threshold": max(0.4, EMBEDDING_MODELS_BENCHMARKS[balanced_components['embedding']].metrics.accuracy - 0.15),
                "metric_type": "COSINE",
                "nprobe": min(256, int(VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.throughput / 200)),
                "rerank_results": True
            },
            "llm": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 1000,
                "context_window": 4096
            },
            "hybrid_search": {
                "enabled": VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.throughput > 20000
            },
            "performance": {
                "batch_size": int(VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.throughput / 2000),
                "max_concurrent": 4,
                "cache_size_mb": int(VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.memory_mb / 8)
            }
        }
        
        pipelines.append(
            EnhancedRAGPipeline(
                name="Standard RAG Pipeline",
                description=f"Balanced RAG pipeline using {balanced_components['embedding']} embeddings and {balanced_components['vector_db']} vector store",
                ingestion_pipeline=balanced_pipeline,
                rag_config=AdvancedRAGConfig(
                    name="Standard RAG",
                    description=f"Well-balanced RAG configuration for {requirements.data_characteristics.primary_data_type.value} and {', '.join(dt.value for dt in requirements.data_characteristics.secondary_data_types)}",
                    configuration=balanced_config
                ),
                when_to_use=f"Optimal for {EMBEDDING_MODELS_BENCHMARKS[balanced_components['embedding']].best_use_cases[0]}",
                trade_offs=[
                    f"Balanced accuracy ({EMBEDDING_MODELS_BENCHMARKS[balanced_components['embedding']].metrics.accuracy:.2f})",
                    f"Good throughput ({VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.throughput} vectors/sec)",
                    f"Memory usage: {VECTOR_DB_BENCHMARKS[balanced_components['vector_db']].metrics.memory_mb}MB",
                    f"Estimated cost: ${EMBEDDING_MODELS_BENCHMARKS[balanced_components['embedding']].metrics.cost_per_1k:.4f}/1K tokens"
                ]
            )
        )

        # Create the Resource-Efficient Pipeline with benchmark-optimized settings
        resource_config = {
            "retrieval": {
                "enabled": True,
                "top_k": max(3, int(3 * VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.accuracy)),
                "score_threshold": max(0.5, EMBEDDING_MODELS_BENCHMARKS[resource_components['embedding']].metrics.accuracy - 0.2),
                "metric_type": "COSINE",
                "nprobe": min(64, int(VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.throughput / 500)),
                "rerank_results": False
            },
            "llm": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 500,
                "context_window": 4096
            },
            "hybrid_search": {
                "enabled": False
            },
            "performance": {
                "batch_size": max(1, int(VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.throughput / 5000)),
                "max_concurrent": 2,
                "cache_size_mb": min(
                    256,
                    int(VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.memory_mb / 16)
                )
            }
        }
        
        pipelines.append(
            EnhancedRAGPipeline(
                name="Lightweight RAG Pipeline",
                description=f"Resource-efficient RAG pipeline using {resource_components['embedding']} embeddings and {resource_components['vector_db']} vector store",
                ingestion_pipeline=resource_pipeline,
                rag_config=AdvancedRAGConfig(
                    name="Basic RAG",
                    description=f"Streamlined RAG configuration optimized for {requirements.data_characteristics.primary_data_type.value} and {', '.join(dt.value for dt in requirements.data_characteristics.secondary_data_types)}",
                    configuration=resource_config
                ),
                when_to_use=f"Optimal for {VECTOR_DB_BENCHMARKS[resource_components['vector_db']].best_use_cases[0]}",
                trade_offs=[
                    f"Lower resource usage (Memory: {VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.memory_mb}MB)",
                    f"Basic features with {EMBEDDING_MODELS_BENCHMARKS[resource_components['embedding']].metrics.accuracy:.2f} accuracy",
                    f"Throughput: {VECTOR_DB_BENCHMARKS[resource_components['vector_db']].metrics.throughput} vectors/sec",
                    f"Lowest cost: ${EMBEDDING_MODELS_BENCHMARKS[resource_components['embedding']].metrics.cost_per_1k:.4f}/1K tokens"
                ]
            )
        )
        return pipelines

    def _analyze_requirements_comprehensively(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> Dict[str, str]:
        """Comprehensive analysis of requirements"""
        data_types = f"{requirements.data_characteristics.primary_data_type.value}, {', '.join(dt.value for dt in requirements.data_characteristics.secondary_data_types)}"
        domain = requirements.data_characteristics.content_domain.value
        complexity = requirements.data_characteristics.document_complexity.value
        volume = requirements.data_characteristics.current_volume.value
        
        data_analysis = (
            f"Processing {volume} volume of {complexity} {data_types} "
            f"in {domain} domain"
        )
        
        use_case = requirements.use_case_requirements.primary_use_case.value
        query_complexity = requirements.use_case_requirements.expected_query_complexity.value
        latency = requirements.use_case_requirements.latency_requirement.value
        
        use_case_analysis = (
            f"{use_case} with {query_complexity} query complexity and "
            f"{latency} latency requirements"
        )
        
        budget = requirements.business_context.budget_range.value
        team = requirements.business_context.team_expertise.value
        
        business_constraints = (
            f"Operating with {budget} budget and {team} team expertise"
        )
        
        scalability = requirements.technical_preferences.scalability_requirements
        monitoring = requirements.technical_preferences.monitoring_depth
        
        technical_implications = (
            f"Requires {scalability} scalability with {monitoring} "
            f"monitoring requirements"
        )
        
        return {
            "data_analysis": data_analysis,
            "use_case_analysis": use_case_analysis,
            "business_constraints": business_constraints,
            "technical_implications": technical_implications
        }

    def _analyze_trade_offs(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> Dict[str, str]:
        """Analyze trade-offs in recommendations"""
        # Cost vs Performance Analysis
        budget = requirements.business_context.budget_range
        latency_req = requirements.use_case_requirements.latency_requirement
        cost_performance = (
            "High performance optimization possible" if budget in [BudgetRange.HIGH, BudgetRange.UNLIMITED]
            else "Performance constrained by budget" if budget in [BudgetRange.MINIMAL, BudgetRange.LOW]
            else "Balanced cost and performance trade-off"
        )

        # Accuracy vs Speed Analysis
        accuracy = requirements.use_case_requirements.accuracy_tolerance
        domain = requirements.data_characteristics.content_domain
        accuracy_speed = (
            "Maximum accuracy with acceptable latency impact" 
            if accuracy == AccuracyTolerance.ZERO_TOLERANCE or domain in [ContentDomain.LEGAL, ContentDomain.MEDICAL]
            else "Speed-optimized with good accuracy" 
            if accuracy == AccuracyTolerance.HIGH_TOLERANCE and latency_req == LatencyRequirement.REAL_TIME
            else "Balanced accuracy and speed"
        )

        # Flexibility vs Complexity
        expertise = requirements.business_context.team_expertise
        flexibility = (
            "High flexibility with managed complexity"
            if expertise in [TeamExpertise.EXPERT, TeamExpertise.ADVANCED]
            else "Simplified configuration with essential flexibility"
            if expertise == TeamExpertise.BASIC_TECHNICAL
            else "Moderate customization with manageable complexity"
        )

        # Scalability vs Maintenance
        scalability = requirements.technical_preferences.scalability_requirements
        volume = requirements.data_characteristics.current_volume
        scale_maintain = (
            f"Optimized for {scalability} scalability with corresponding maintenance needs"
            if volume in [VolumeSize.LARGE, VolumeSize.VERY_LARGE, VolumeSize.MASSIVE]
            else "Balanced scaling and maintenance requirements"
        )

        return {
            "cost_vs_performance": cost_performance,
            "accuracy_vs_speed": accuracy_speed,
            "flexibility_vs_complexity": flexibility,
            "scalability_vs_maintenance": scale_maintain
        }

    def _assess_risks(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> Dict[str, str]:
        """Assess implementation and operational risks"""
        risks = {}
        
        # Implementation Risks
        team_expertise = requirements.business_context.team_expertise
        impl_risks = []
        
        if team_expertise == TeamExpertise.BASIC_TECHNICAL:
            impl_risks.append("Limited team expertise may require additional training")
        
        risks["implementation_risks"] = (
            " and ".join(impl_risks) if impl_risks 
            else "Standard implementation risks, manageable with proper planning"
        )
        
        # Operational Risks
        domain = requirements.data_characteristics.content_domain
        volume = requirements.data_characteristics.current_volume
        latency = requirements.use_case_requirements.latency_requirement
        op_risks = []
        
        if domain in [ContentDomain.LEGAL, ContentDomain.MEDICAL]:
            op_risks.append(f"Critical {domain.value} domain requires strict data handling")
        if volume in [VolumeSize.VERY_LARGE, VolumeSize.MASSIVE]:
            op_risks.append("High data volume requires robust scaling strategy")
        if latency == LatencyRequirement.REAL_TIME:
            op_risks.append("Real-time requirements need careful performance monitoring")
            
        risks["operational_risks"] = (
            " and ".join(op_risks) if op_risks
            else "Normal operational challenges, can be addressed with monitoring"
        )
        
        # Technical Risks
        tech_risks = []
        if requirements.data_characteristics.document_complexity.value == "highly_complex":
            tech_risks.append("Complex document processing may require specialized handling")
            
        risks["technical_risks"] = (
            " and ".join(tech_risks) if tech_risks
            else "Standard technical risks, mitigatable with best practices"
        )
        
        # Business Risks
        budget = requirements.business_context.budget_range
        accuracy = requirements.use_case_requirements.accuracy_tolerance
        bus_risks = []
        
        if budget in [BudgetRange.MINIMAL, BudgetRange.LOW]:
            bus_risks.append("Budget constraints may limit optimal solution choices")
        if accuracy == AccuracyTolerance.ZERO_TOLERANCE:
            bus_risks.append("Zero tolerance for errors requires premium components")
            
        risks["business_risks"] = (
            " and ".join(bus_risks) if bus_risks
            else "Limited business risks with proposed solution"
        )
        
        return risks

    def _create_implementation_roadmap(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> List[str]:
        """Create step-by-step implementation roadmap"""
        roadmap = []
        
        # Initial Assessment Phase
        domain = requirements.data_characteristics.content_domain
        if domain in [ContentDomain.LEGAL, ContentDomain.MEDICAL, ContentDomain.FINANCIAL]:
            roadmap.append(f"Phase 1: Domain Analysis and Compliance Review for {domain.value}")
        else:
            roadmap.append("Phase 1: Requirements Analysis and Planning")
            
        # Infrastructure Setup
        roadmap.append("Phase 2: Infrastructure Setup and Configuration")
            
        # Data Processing Setup
        complexity = requirements.data_characteristics.document_complexity
        volume = requirements.data_characteristics.current_volume
        if complexity.value in ["complex", "highly_complex"] or volume in [VolumeSize.VERY_LARGE, VolumeSize.MASSIVE]:
            roadmap.append("Phase 3: Advanced Data Processing Pipeline Implementation")
            roadmap.append("Phase 4: Chunking and Embedding Optimization")
        else:
            roadmap.append("Phase 3: Data Processing Pipeline Setup")
            
        # RAG Setup
        accuracy = requirements.use_case_requirements.accuracy_tolerance
        if accuracy == AccuracyTolerance.ZERO_TOLERANCE:
            roadmap.append("Phase 5: High-Precision RAG Implementation")
            roadmap.append("Phase 6: Accuracy Validation and Fine-tuning")
        else:
            roadmap.append("Phase 5: RAG Integration and Testing")
            
        # Testing & Monitoring
        # Check if monitoring depth is comprehensive or standard
        if requirements.technical_preferences.monitoring_depth == "comprehensive":
            roadmap.append("Phase 7: Comprehensive Monitoring Setup")
            roadmap.append("Phase 8: Performance Optimization and Tuning")
        else:
            roadmap.append("Phase 7: Basic Monitoring and Deployment")
            
        return roadmap

    def _define_success_metrics(
        self,
        requirements: ComprehensiveUserRequirements
    ) -> List[str]:
        """Define measurable success metrics"""
        return [
            "Query response accuracy > 90%",
            "Average response time < 3s",
            "System uptime > 99.9%",
            "User satisfaction score > 4.5/5"
        ]

    def _create_monitoring_recommendations(
        self,
        requirements: ComprehensiveUserRequirements
    ) -> List[str]:
        """Create monitoring recommendations"""
        return [
            "Monitor system performance metrics",
            "Track user feedback and satisfaction",
            "Measure response times and accuracy",
            "Review and optimize resource usage"
        ]

    def _generate_alternative_considerations(
        self,
        requirements: ComprehensiveUserRequirements,
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate alternative approaches and considerations"""
        return [
            "Consider hybrid search strategies",
            "Evaluate alternative embedding models",
            "Explore different chunking strategies",
            "Assess alternative vector databases"
        ]

    def _create_evolution_path(
        self,
        requirements: ComprehensiveUserRequirements
    ) -> List[str]:
        """Create future evolution recommendations"""
        return [
            "Phase 1: Initial deployment and stabilization",
            "Phase 2: Performance optimization and tuning",
            "Phase 3: Feature expansion and enhancement",
            "Phase 4: Scale and integrate with other systems"
        ]
        
    def _analyze_performance_metrics(self, pipeline: EnhancedRAGPipeline, config: Dict[str, Any]) -> Dict[str, str]:
        """Analyze detailed performance metrics from benchmarks"""
        insights = config.get("benchmark_insights", {})
        
        performance = {}
        
        if "embedding_model" in insights:
            emb = insights["embedding_model"]
            performance["embedding_throughput"] = f"Embedding Generation: {emb['throughput']}"
            performance["embedding_latency"] = f"Average Latency: {emb['latency']}"
            performance["embedding_accuracy"] = f"Embedding Quality: {emb['accuracy']}"
            
        if "vector_db" in insights:
            vdb = insights["vector_db"]
            performance["indexing_speed"] = f"Vector Indexing: {vdb['throughput']}"
            performance["query_latency"] = f"Query Latency (p95): {vdb['latency']}"
            performance["search_accuracy"] = f"Search Accuracy: {vdb['accuracy']}"
            
        if "chunking" in insights:
            chunk = insights["chunking"]
            performance["chunking_speed"] = f"Document Processing: {chunk['throughput']}"
            performance["chunking_quality"] = f"Chunking Quality: {chunk['accuracy']}"
            
        return performance
        
    def _analyze_resource_requirements(self, pipeline: EnhancedRAGPipeline, config: Dict[str, Any]) -> Dict[str, str]:
        """Analyze resource requirements based on benchmarks"""
        insights = config.get("benchmark_insights", {})
        
        resources = {}
        total_memory = 0
        
        if "embedding_model" in insights:
            mem = int(insights["embedding_model"]["memory_usage"].replace("MB", ""))
            total_memory += mem
            resources["embedding_memory"] = f"Embedding Model: {mem}MB RAM"
            
        if "vector_db" in insights:
            mem = int(insights["vector_db"]["memory_usage"].replace("MB", ""))
            total_memory += mem
            resources["vector_db_memory"] = f"Vector Database: {mem}MB RAM"
            
        if "chunking" in insights:
            mem = int(insights["chunking"]["memory_usage"].replace("MB", ""))
            total_memory += mem
            resources["chunking_memory"] = f"Text Processing: {mem}MB RAM"
            
        resources["total_memory"] = f"Total Memory Required: {total_memory}MB RAM"
        resources["cpu_cores"] = f"Recommended CPU Cores: {config['embed_config']['optimization']['num_workers']}"
        resources["gpu_required"] = f"GPU Acceleration: {'Recommended' if config['embed_config']['optimization']['use_gpu'] else 'Optional'}"
        
        return resources
        
    def _analyze_cost_metrics(self, pipeline: EnhancedRAGPipeline, config: Dict[str, Any]) -> Dict[str, str]:
        """Analyze cost metrics from benchmarks"""
        insights = config.get("benchmark_insights", {})
        
        costs = {}
        
        if "embedding_model" in insights:
            costs["embedding_cost"] = f"Embedding Generation: {insights['embedding_model']['cost']}"
            
        if "vector_db" in insights:
            costs["storage_cost"] = f"Vector Storage: {insights['vector_db']['cost']}"
            
        if "chunking" in insights:
            costs["processing_cost"] = f"Text Processing: {insights['chunking']['cost']}"
            
        # Calculate estimated monthly cost for 100GB data
        total_cost = sum(float(x["cost"].replace("$", "").split("/")[0]) 
                        for x in insights.values())
        costs["monthly_estimate"] = f"Estimated Monthly Cost (100GB): ${total_cost*100:.2f}"
        
        return costs
        
    def _analyze_scaling_characteristics(self, pipeline: EnhancedRAGPipeline, config: Dict[str, Any]) -> Dict[str, str]:
        """Analyze scaling characteristics"""
        insights = config.get("benchmark_insights", {})
        scaling = {}
        
        # Analyze throughput scalability
        if "vector_db" in insights:
            vdb = insights["vector_db"]
            max_throughput = int(vdb["throughput"].split()[0])
            scaling["max_throughput"] = f"Maximum Throughput: {max_throughput:,} vectors/second"
            scaling["concurrent_queries"] = f"Concurrent Queries: {max_throughput//100} recommended"
            
        # Document processing capacity
        if "chunking" in insights:
            chunk = insights["chunking"]
            docs_per_sec = int(chunk["throughput"].split()[0])
            scaling["ingestion_rate"] = f"Document Ingestion: {docs_per_sec:,} documents/second"
            scaling["daily_capacity"] = f"Daily Processing Capacity: {docs_per_sec * 86400:,} documents"
            
        # Memory scaling
        total_memory = sum(int(x["memory_usage"].replace("MB", "")) 
                         for x in insights.values())
        per_million_docs = total_memory * 2  # Estimate for 1M documents
        scaling["memory_scaling"] = f"Memory per Million Documents: {per_million_docs}MB"
        
        return scaling
