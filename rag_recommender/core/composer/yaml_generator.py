"""
YAML configuration generator for RAG pipeline recommendations.
"""
import yaml
from typing import Dict, Any
from ..models.base import EnhancedRAGPipeline
from ..benchmarks.model_benchmarks import (
    EMBEDDING_MODELS_BENCHMARKS,
    VECTOR_DB_BENCHMARKS,
    CHUNKING_BENCHMARKS
)

def generate_yaml_config(pipeline: EnhancedRAGPipeline, data_type: str = "pdf") -> Dict[str, Any]:
    """Generate a detailed YAML configuration for a recommended pipeline."""
    
    # Get benchmark details for components
    embedding_benchmark = EMBEDDING_MODELS_BENCHMARKS.get(pipeline.ingestion_pipeline.embedding.model_name)
    vector_db_benchmark = VECTOR_DB_BENCHMARKS.get(pipeline.ingestion_pipeline.vector_db.database_type)
    chunking_benchmark = CHUNKING_BENCHMARKS.get(pipeline.ingestion_pipeline.chunking.chunking_strategy)
    
    config = {
        "data_source": {
            "type": data_type,
            "file": f"data/{data_type}_files/*",
            f"{data_type}_config": {
                "extract_tables": True,
                "include_metadata": True,
                "ocr_enabled": data_type == "pdf"
            }
        },
        "embed_config": {
            "model": pipeline.ingestion_pipeline.embedding.model_name,
            "batch_size": embedding_benchmark.metrics.throughput // 10 if embedding_benchmark else 256,
            "normalize": True,
            "dimension": embedding_benchmark.metrics.memory_mb // 4 if embedding_benchmark else 384,
            "embedding_type": "semantic",
            "optimization": {
                "use_gpu": pipeline.rag_config.configuration.get("performance", {}).get("use_gpu", True),
                "half_precision": True,
                "num_workers": pipeline.rag_config.configuration.get("performance", {}).get("max_concurrent", 8)
            },
            "dimension_reduction": {
                "use_pca": pipeline.ingestion_pipeline.embedding.configuration.get("use_pca", True),
                "random_state": 42,
                "whiten": False
            }
        },
        "chunking": {
            "strategy": pipeline.ingestion_pipeline.chunking.chunking_strategy,
            "chunk_size": pipeline.ingestion_pipeline.chunking.chunk_size,
            "overlap": pipeline.ingestion_pipeline.chunking.chunk_overlap,
            "delimiter": "\\n\\n",
            "combine_small_chunks": True,
            "min_chunk_size": pipeline.ingestion_pipeline.chunking.configuration.get("min_chunk_size", 100),
            "sentence_options": {
                "language": "english",
                "respect_breaks": True
            },
            "advanced": {
                "normalize_whitespace": True,
                "preserve_line_breaks": False,
                "smart_splitting": pipeline.ingestion_pipeline.chunking.configuration.get("smart_splitting", True)
            }
        },
        "vector_db": {
            "type": pipeline.ingestion_pipeline.vector_db.database_type,
            "host": "localhost",
            "port": 6333 if pipeline.ingestion_pipeline.vector_db.database_type == "qdrant" else 19530,
            "collection": f"{data_type}_embeddings",
            "connection_retries": 3,
            "retry_delay": 5,
            "timeout": pipeline.rag_config.configuration.get("performance", {}).get("timeout", 300),
            "batch_size": pipeline.rag_config.configuration.get("performance", {}).get("batch_size", 300),
            "vector_index": {
                "type": "hnsw",
                "params": {
                    "m": 16,
                    "ef_construct": vector_db_benchmark.metrics.throughput // 200 if vector_db_benchmark else 256,
                    "ef_runtime": vector_db_benchmark.metrics.throughput // 400 if vector_db_benchmark else 128,
                    "full_scan_threshold": vector_db_benchmark.metrics.throughput // 5 if vector_db_benchmark else 10000
                },
                "quantization": {
                    "enabled": pipeline.rag_config.configuration.get("performance", {}).get("use_quantization", True),
                    "always_ram": True,
                    "quantum": 0.01
                }
            },
            "payload_index": [
                {"field": "page_number", "type": "integer"},
                {"field": "chunk_index", "type": "integer"},
                {"field": "total_chunks", "type": "integer"},
                {"field": "chunk_size", "type": "integer"},
                {"field": "chunking_strategy", "type": "keyword"},
                {"field": "content_type", "type": "keyword"},
                {"field": "metadata", "type": "object"}
            ]
        },
        "retrieval": {
            "enabled": True,
            "top_k": pipeline.rag_config.configuration.get("retrieval", {}).get("top_k", 10),
            "score_threshold": pipeline.rag_config.configuration.get("retrieval", {}).get("score_threshold", 0.4),
            "metric_type": "COSINE",
            "nprobe": pipeline.rag_config.configuration.get("retrieval", {}).get("nprobe", 256),
            "rerank_results": pipeline.rag_config.configuration.get("retrieval", {}).get("rerank_results", True),
            "diversity_weight": pipeline.rag_config.configuration.get("retrieval", {}).get("diversity_weight", 0.2)
        }
    }
    
    # Add benchmark-based insights
    if embedding_benchmark:
        config["benchmark_insights"] = {
            "embedding_model": {
                "throughput": f"{embedding_benchmark.metrics.throughput} tokens/sec",
                "latency": f"{embedding_benchmark.metrics.latency_ms}ms average",
                "memory_usage": f"{embedding_benchmark.metrics.memory_mb}MB",
                "accuracy": f"{embedding_benchmark.metrics.accuracy:.4f}",
                "cost": f"${embedding_benchmark.metrics.cost_per_1k}/1K tokens",
                "best_use_cases": embedding_benchmark.best_use_cases,
                "limitations": embedding_benchmark.limitations
            }
        }
        
        if vector_db_benchmark:
            config["benchmark_insights"]["vector_db"] = {
                "throughput": f"{vector_db_benchmark.metrics.throughput} vectors/sec",
                "latency": f"{vector_db_benchmark.metrics.latency_ms}ms p95",
                "memory_usage": f"{vector_db_benchmark.metrics.memory_mb}MB",
                "accuracy": f"{vector_db_benchmark.metrics.accuracy:.4f}",
                "cost": f"${vector_db_benchmark.metrics.cost_per_1k}/1K ops",
                "best_use_cases": vector_db_benchmark.best_use_cases,
                "limitations": vector_db_benchmark.limitations
            }
            
        if chunking_benchmark:
            config["benchmark_insights"]["chunking"] = {
                "throughput": f"{chunking_benchmark.metrics.throughput} docs/sec",
                "latency": f"{chunking_benchmark.metrics.latency_ms}ms per doc",
                "memory_usage": f"{chunking_benchmark.metrics.memory_mb}MB",
                "accuracy": f"{chunking_benchmark.metrics.accuracy:.4f}",
                "cost": f"${chunking_benchmark.metrics.cost_per_1k}/1K chunks",
                "best_use_cases": chunking_benchmark.best_use_cases,
                "limitations": chunking_benchmark.limitations
            }
    
    return config

def save_yaml_config(config: Dict[str, Any], filepath: str) -> None:
    """Save the configuration to a YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
