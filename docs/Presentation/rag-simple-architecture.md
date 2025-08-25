# RAG Pipeline Architecture - Simplified Container-Based System

## 1. System Architecture Overview

### 1.1 Simplified Architectural Layers

**Layer 1: User Interface Layer**
- **Frontend Web Application**: React-based UI for user interaction
- **Use Case Forms**: Dynamic requirement collection interface
- **File Upload Interface**: Drag-and-drop for user data files
- **Pipeline Dashboard**: Interactive recommendation display and selection
- **Progress Tracking**: Real-time optimization status updates

**Layer 2: Data Processing Layer**
- **DASK Scheduler**: Central task coordinator running in container
- **DASK Workers**: Parallel processing containers (auto-scaling)
- **Document Processor**: Handles file parsing and text extraction
- **Embedding Analyzer**: Generates sample embeddings for evaluation
- **Statistics Calculator**: Computes data quality metrics

**Layer 3: Orchestration Layer**
- **Pipeline Orchestrator**: Central coordination service (FastAPI)
- **Benchmark Manager**: External data integration module
- **Optimization Engine**: Hierarchical MAB + Bayesian optimizer
- **Configuration Manager**: Pipeline configuration storage and retrieval
- **Recommendation Generator**: Pareto-optimal solution generator

**Layer 4: Core RAG Services** 
- **Embedding Service**: Multi-model embedding API
- **Vector Database Service**: Unified vector search interface
- **LLM Inference Service**: Language model processing
- **Re-ranking Service**: Result refinement and ordering

**Layer 5: Data Storage Layer**
- **Local File Storage**: User uploaded documents (temporary)
- **Benchmark Data Store**: Cached external performance data
- **Configuration Database**: Pipeline configs and user preferences  
- **Results Cache**: Optimization results and recommendations
- **Vector Databases**: Multiple database backends for testing

## 2. End-to-End Benchmark Optimization Flow

### 2.1 Benchmark Data Collection and Processing

**Step 1: External Benchmark Data Sources**
```yaml
benchmark_sources:
  academic_benchmarks:
    mteb_leaderboard:
      url: "https://huggingface.co/spaces/mteb/leaderboard" 
      data_points: ["model_name", "accuracy", "speed", "memory_usage"]
      update_frequency: "daily"
      
    beir_benchmark:
      url: "https://github.com/beir-cellar/beir/results"
      data_points: ["model", "dataset", "ndcg", "recall", "map"]
      update_frequency: "weekly"
      
  industry_benchmarks:
    pinecone_reports:
      source: "Pinecone Performance Studies"
      data_points: ["qps", "latency", "cost_per_query", "accuracy"]
      domains: ["e-commerce", "support", "legal", "healthcare"]
      
    openai_benchmarks:
      source: "OpenAI Model Cards"
      data_points: ["model", "cost_per_token", "context_length", "speed"]
      models: ["gpt-4o", "gpt-4o-mini", "text-embedding-3"]
```

**Step 2: Benchmark Data Processing**
```python
class BenchmarkProcessor:
    def process_benchmark_data(self, raw_data: Dict) -> List[PipelineArchetype]:
        # 1. Clean and normalize benchmark data
        cleaned_data = self.clean_benchmark_data(raw_data)
        
        # 2. Extract performance configurations
        configurations = []
        for benchmark in cleaned_data:
            config = {
                'embedding_model': benchmark.model,
                'accuracy': benchmark.score,
                'latency': benchmark.response_time,
                'cost': benchmark.cost_per_query,
                'use_case': benchmark.domain
            }
            configurations.append(config)
            
        # 3. Cluster configurations into archetypes
        clusters = self.cluster_configurations(configurations)
        
        # 4. Create archetype templates
        archetypes = []
        for cluster in clusters:
            archetype = self.create_archetype_from_cluster(cluster)
            archetypes.append(archetype)
            
        return archetypes

    def create_archetype_from_cluster(self, cluster: List[Dict]) -> PipelineArchetype:
        # Calculate average performance metrics for cluster
        avg_accuracy = np.mean([config['accuracy'] for config in cluster])
        avg_latency = np.mean([config['latency'] for config in cluster])
        avg_cost = np.mean([config['cost'] for config in cluster])
        
        # Determine dominant configuration pattern
        common_embedding = self.most_common([config['embedding_model'] for config in cluster])
        
        return PipelineArchetype(
            name=f"archetype_{len(self.archetypes)}",
            embedding_model=common_embedding,
            expected_accuracy=avg_accuracy,
            expected_latency=avg_latency,
            expected_cost=avg_cost,
            use_cases=self.extract_use_cases(cluster),
            confidence_score=self.calculate_confidence(cluster)
        )
```

### 2.2 User Requirements Processing

**Step 3: User Input Analysis**
```python
class UserRequirementsAnalyzer:
    def analyze_user_requirements(self, requirements: UserRequirements) -> RequirementProfile:
        profile = RequirementProfile()
        
        # Analyze use case priority
        profile.use_case_category = self.classify_use_case(requirements.use_case)
        profile.volume_category = self.classify_volume(requirements.daily_queries)
        
        # Create priority weights
        total_priority = requirements.accuracy_priority + requirements.speed_priority + requirements.cost_priority
        profile.accuracy_weight = requirements.accuracy_priority / total_priority
        profile.speed_weight = requirements.speed_priority / total_priority  
        profile.cost_weight = requirements.cost_priority / total_priority
        
        # Set hard constraints
        profile.max_latency = requirements.max_acceptable_latency
        profile.max_cost = requirements.max_cost_per_query
        profile.min_accuracy = requirements.min_acceptable_accuracy
        
        return profile
```

### 2.3 DASK User Data Processing

**Step 4: Distributed Data Analysis**
```python
class DaskDataProcessor:
    def __init__(self, scheduler_address: str):
        self.client = Client(scheduler_address)
        
    def process_user_files(self, file_paths: List[str]) -> DataProfile:
        # Submit parallel processing tasks
        processing_futures = []
        
        for file_path in file_paths:
            # Parse document in parallel
            parse_future = self.client.submit(self.parse_document, file_path)
            processing_futures.append(parse_future)
            
        # Wait for parsing to complete
        documents = self.client.gather(processing_futures)
        
        # Parallel analysis tasks
        analysis_tasks = {
            'statistics': self.client.submit(self.compute_statistics, documents),
            'embeddings': self.client.submit(self.generate_sample_embeddings, documents),
            'complexity': self.client.submit(self.analyze_complexity, documents),
            'domain': self.client.submit(self.classify_domain, documents)
        }
        
        # Gather all results
        results = self.client.gather(list(analysis_tasks.values()))
        
        return DataProfile(
            document_count=len(documents),
            avg_length=results[0]['avg_length'],
            vocabulary_size=results[0]['vocab_size'],
            sample_embeddings=results[1],
            complexity_score=results[2],
            domain_classification=results[3],
            processing_time=time.time() - start_time
        )
        
    def generate_test_queries(self, documents: List[str]) -> List[str]:
        # Generate representative queries from document content
        query_futures = []
        
        for doc in documents[:10]:  # Sample first 10 documents
            future = self.client.submit(self.extract_queries_from_doc, doc)
            query_futures.append(future)
            
        all_queries = self.client.gather(query_futures)
        return [query for queries in all_queries for query in queries]  # Flatten
```

### 2.4 Hierarchical MAB Optimization

**Step 5: High-Level Archetype Selection**
```python
class HierarchicalMAB:
    def __init__(self, archetypes: List[PipelineArchetype]):
        self.archetypes = archetypes
        self.archetype_rewards = {arch.name: [] for arch in archetypes}
        self.selection_counts = {arch.name: 0 for arch in archetypes}
        
    def select_archetype(self, user_profile: RequirementProfile) -> PipelineArchetype:
        # Calculate UCB1 scores for each archetype
        ucb_scores = {}
        total_selections = sum(self.selection_counts.values())
        
        for archetype in self.archetypes:
            name = archetype.name
            
            if self.selection_counts[name] == 0:
                # Unselected archetypes get highest priority
                ucb_scores[name] = float('inf')
            else:
                # Calculate UCB1 score
                avg_reward = np.mean(self.archetype_rewards[name])
                confidence_bound = np.sqrt(2 * np.log(total_selections) / self.selection_counts[name])
                
                # Weight by user requirements compatibility
                compatibility_score = self.calculate_compatibility(archetype, user_profile)
                
                ucb_scores[name] = avg_reward + confidence_bound + compatibility_score
                
        # Select archetype with highest UCB1 score
        selected_name = max(ucb_scores.keys(), key=lambda x: ucb_scores[x])
        selected_archetype = next(arch for arch in self.archetypes if arch.name == selected_name)
        
        self.selection_counts[selected_name] += 1
        return selected_archetype
        
    def calculate_compatibility(self, archetype: PipelineArchetype, profile: RequirementProfile) -> float:
        # Calculate how well archetype matches user requirements
        accuracy_match = 1.0 - abs(archetype.expected_accuracy - profile.target_accuracy) 
        latency_match = 1.0 - abs(archetype.expected_latency - profile.target_latency) / profile.max_latency
        cost_match = 1.0 - abs(archetype.expected_cost - profile.target_cost) / profile.max_cost
        
        # Weight by user priorities
        compatibility = (
            profile.accuracy_weight * accuracy_match +
            profile.speed_weight * latency_match + 
            profile.cost_weight * cost_match
        )
        
        return compatibility
```

**Step 6: Low-Level Parameter Optimization**
```python
class ParameterMAB:
    def __init__(self, archetype: PipelineArchetype):
        self.archetype = archetype
        self.parameter_space = self.define_parameter_space(archetype)
        self.parameter_rewards = {}
        
    def optimize_parameters(self, user_data: DataProfile) -> Dict:
        optimized_params = {}
        
        for param_name, param_range in self.parameter_space.items():
            # Use contextual bandit for each parameter
            if param_name == 'chunk_size':
                optimized_params[param_name] = self.optimize_chunk_size(user_data)
            elif param_name == 'top_k':
                optimized_params[param_name] = self.optimize_top_k(user_data)
            elif param_name == 'embedding_model':
                optimized_params[param_name] = self.select_embedding_model(user_data)
                
        return optimized_params
        
    def optimize_chunk_size(self, user_data: DataProfile) -> int:
        # Consider document characteristics for chunk size
        avg_doc_length = user_data.avg_length
        complexity = user_data.complexity_score
        
        # Use Thompson sampling for chunk size selection
        chunk_options = [128, 256, 512, 1024]
        
        # Select based on document characteristics and past performance
        if avg_doc_length < 500:
            preferred_chunks = [128, 256]
        elif avg_doc_length < 2000:
            preferred_chunks = [256, 512]
        else:
            preferred_chunks = [512, 1024]
            
        # Thompson sampling from preferred options
        return self.thompson_sample(preferred_chunks, 'chunk_size')
```

### 2.5 Bayesian Optimization for Continuous Parameters

**Step 7: Multi-Objective Bayesian Optimization**
```python
class BayesianOptimizer:
    def __init__(self, objective_functions: List[Callable]):
        self.objectives = objective_functions
        self.gp_models = [GaussianProcess() for _ in objective_functions]
        self.evaluated_points = []
        self.objective_values = []
        
    def optimize(self, parameter_space: Dict, n_iterations: int = 50) -> List[Dict]:
        for iteration in range(n_iterations):
            # Acquisition function balances exploration and exploitation
            next_point = self.acquisition_function(parameter_space)
            
            # Evaluate objectives at the next point
            pipeline_config = self.point_to_config(next_point)
            objective_scores = self.evaluate_pipeline(pipeline_config)
            
            # Update Gaussian Process models
            self.update_models(next_point, objective_scores)
            
            # Store results
            self.evaluated_points.append(next_point)
            self.objective_values.append(objective_scores)
            
        # Find Pareto front
        pareto_optimal = self.find_pareto_front(self.evaluated_points, self.objective_values)
        return [self.point_to_config(point) for point in pareto_optimal]
        
    def evaluate_pipeline(self, config: Dict) -> List[float]:
        # Evaluate pipeline configuration on user data
        accuracy = self.evaluate_accuracy(config)
        latency = self.evaluate_latency(config)
        cost = self.evaluate_cost(config)
        
        # Return objective values (higher is better, so negate latency and cost)
        return [accuracy, -latency, -cost]
```

### 2.6 Pipeline Evaluation and Recommendation

**Step 8: Real-time Pipeline Evaluation**
```python
class PipelineEvaluator:
    def __init__(self, dask_client: Client):
        self.dask_client = dask_client
        
    def evaluate_pipeline_on_user_data(self, config: PipelineConfig, user_data: DataProfile) -> EvaluationResults:
        # Generate test queries from user documents
        test_queries = user_data.sample_queries[:20]  # Use sample for evaluation
        
        # Parallel evaluation across queries
        evaluation_futures = []
        for query in test_queries:
            future = self.dask_client.submit(self.evaluate_single_query, query, config)
            evaluation_futures.append(future)
            
        # Gather results
        query_results = self.dask_client.gather(evaluation_futures)
        
        # Aggregate metrics
        accuracy_scores = [r.accuracy for r in query_results]
        latency_scores = [r.latency for r in query_results]
        cost_scores = [r.cost for r in query_results]
        
        return EvaluationResults(
            accuracy=np.mean(accuracy_scores),
            accuracy_std=np.std(accuracy_scores),
            latency=np.mean(latency_scores), 
            latency_p95=np.percentile(latency_scores, 95),
            cost_per_query=np.mean(cost_scores),
            total_evaluation_cost=sum(cost_scores),
            confidence_interval=self.calculate_confidence_interval(accuracy_scores)
        )
        
    def evaluate_single_query(self, query: str, config: PipelineConfig) -> QueryResult:
        start_time = time.time()
        
        # Run through pipeline configuration
        embedding = self.get_embedding(query, config.embedding_model)
        search_results = self.vector_search(embedding, config.vector_db, config.top_k)
        
        if config.use_reranking:
            search_results = self.rerank_results(query, search_results, config.reranker)
            
        response = self.generate_response(query, search_results, config.llm_model)
        
        latency = time.time() - start_time
        cost = self.calculate_query_cost(config, query, response)
        accuracy = self.calculate_accuracy_score(query, response, search_results)
        
        return QueryResult(accuracy=accuracy, latency=latency, cost=cost)
```

**Step 9: Recommendation Generation and Ranking**
```python
class RecommendationGenerator:
    def generate_recommendations(self, 
                               pareto_configs: List[PipelineConfig],
                               user_profile: RequirementProfile,
                               evaluation_results: List[EvaluationResults]) -> List[PipelineRecommendation]:
        
        recommendations = []
        
        for i, (config, eval_result) in enumerate(zip(pareto_configs, evaluation_results)):
            # Calculate user preference score
            user_score = self.calculate_user_preference_score(eval_result, user_profile)
            
            # Find matching benchmark data
            benchmark_comparison = self.find_benchmark_comparison(config)
            
            # Create recommendation
            rec = PipelineRecommendation(
                id=f"pipeline_{i}",
                name=self.generate_pipeline_name(config),
                configuration=config,
                performance=eval_result,
                user_preference_score=user_score,
                benchmark_comparison=benchmark_comparison,
                confidence_score=self.calculate_confidence(eval_result),
                trade_offs=self.analyze_trade_offs(eval_result, user_profile)
            )
            
            recommendations.append(rec)
            
        # Sort by user preference score
        recommendations.sort(key=lambda x: x.user_preference_score, reverse=True)
        
        # Add diversity to top recommendations
        diverse_recommendations = self.ensure_diversity(recommendations[:10])
        
        return diverse_recommendations[:5]  # Return top 5 diverse recommendations
```

## 3. Container-Based Deployment

### 3.1 Docker Compose Configuration

```yaml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://orchestrator:8000
    depends_on:
      - orchestrator
      
  orchestrator:
    build: ./orchestrator
    ports:
      - "8000:8000"
    environment:
      - DASK_SCHEDULER_ADDRESS=dask-scheduler:8786
      - DATABASE_URL=postgresql://user:password@postgres:5432/rag_db
    depends_on:
      - dask-scheduler
      - postgres
      - redis
      
  dask-scheduler:
    image: daskdev/dask:latest
    command: ["dask-scheduler"]
    ports:
      - "8786:8786"
      - "8787:8787"
    environment:
      - DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=True
      
  dask-worker:
    image: daskdev/dask:latest
    command: ["dask-worker", "dask-scheduler:8786", "--nthreads", "4", "--memory-limit", "4GB"]
    deploy:
      replicas: 3
    depends_on:
      - dask-scheduler
    environment:
      - DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.7
      
  embedding-service:
    build: ./embedding-service
    ports:
      - "8001:8000"
    environment:
      - MODEL_CACHE_DIR=/models
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              
  vector-service:
    build: ./vector-service  
    ports:
      - "8002:8000"
    environment:
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - WEAVIATE_URL=${WEAVIATE_URL}
    depends_on:
      - postgres
      
  llm-service:
    build: ./llm-service
    ports:
      - "8003:8000" 
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru

volumes:
  postgres_data:
```

### 3.2 Service Resource Allocation

```yaml
resource_limits:
  frontend:
    cpu_limit: "1"
    memory_limit: "512MB"
    
  orchestrator:
    cpu_limit: "2"
    memory_limit: "2GB"
    
  dask_scheduler:
    cpu_limit: "1"
    memory_limit: "1GB"
    
  dask_worker:
    cpu_limit: "4"
    memory_limit: "4GB"
    replicas: 3-8  # Auto-scale based on load
    
  embedding_service:
    cpu_limit: "4"
    memory_limit: "6GB"
    gpu_required: true  # Optional, for acceleration
    
  vector_service:
    cpu_limit: "2"
    memory_limit: "4GB"
    
  llm_service:
    cpu_limit: "2"
    memory_limit: "4GB"
```

## 4. Performance Targets and Monitoring

### 4.1 System Performance Targets

**User Interface Response Times:**
- Form submission: <500ms
- File upload (100MB): <30 seconds with progress
- Recommendation display: <1 second
- Pipeline selection: <200ms

**DASK Processing Performance:**
- Document processing: 50MB/minute per worker
- Sample embedding generation: 500 embeddings/minute per worker
- Statistical analysis: Complete within 2 minutes for <1GB data
- Parallel efficiency: >80% CPU utilization during processing

**Optimization Performance:**
- Benchmark data collection: <5 minutes (cached daily)
- Archetype selection: <10 seconds
- Parameter optimization: 20-50 iterations in <10 minutes
- Pipeline evaluation: <5 minutes for 20 test queries
- Total optimization time: <15 minutes for typical workload

### 4.2 Accuracy and Quality Metrics

**Recommendation Quality:**
- User satisfaction with top recommendation: >80%
- Pareto front coverage: >90% of optimal solutions found
- Benchmark accuracy: Within 5% of reported benchmark scores
- Confidence interval coverage: 95% actual performance within predicted range

**System Reliability:**
- Container uptime: >99% availability
- DASK job completion rate: >95%
- External API call success rate: >98%
- Data processing error rate: <1%

This simplified architecture provides a clear, container-based deployment model with comprehensive end-to-end benchmark optimization flow. The system is designed to be easily deployable and maintainable while providing sophisticated pipeline optimization capabilities.