# Create detailed evaluation examples for both RAG ingestion and retrieval
ingestion_evaluation_example = {
    "rag_ingestion_evaluation": {
        "scenario": {
            "domain": "Financial Services Documentation",
            "data_volume": "10,000 PDF documents, 500MB total",
            "use_case": "Regulatory compliance Q&A system",
            "requirements": "High accuracy, moderate latency tolerance"
        },
        
        "user_inputs": {
            "data_sources": ["PDF regulatory documents", "Financial reports", "Compliance guidelines"],
            "sample_data": "100 representative documents provided",
            "use_case_details": {
                "domain": "Financial Services",
                "query_types": ["Factual Q&A", "Compliance verification", "Document search"],
                "complexity": "Moderate to complex"
            },
            "volume_specs": {
                "current_volume": "10,000 documents",
                "growth_rate": "5% monthly",
                "peak_users": "100 concurrent"
            },
            "priorities": "Accuracy (90%) > Cost (60%) > Latency (70%)",
            "constraints": {
                "budget": "$2,000/month",
                "privacy": "Confidential data",
                "compliance": "SOX, SEC regulations"
            }
        },
        
        "evaluation_process": {
            "step_1_baseline": {
                "chunking": {
                    "strategy": "Fixed-size chunking",
                    "parameters": {"chunk_size": 1024, "overlap": 20},
                    "performance": {"processing_speed": "50 docs/min", "context_preservation": 0.75}
                },
                "embedding": {
                    "model": "sentence-transformers/all-mpnet-base-v2", 
                    "dimensions": 768,
                    "performance": {"encoding_speed": "100 chunks/sec", "domain_relevance": 0.72}
                },
                "storage": {
                    "vector_db": "FAISS",
                    "index_type": "Flat",
                    "performance": {"indexing_time": "2 hours", "storage_size": "1.2GB"}
                }
            },
            
            "step_2_optimization": {
                "chunking_optimization": {
                    "tested_strategies": [
                        {"strategy": "Fixed-size", "chunk_size": 512, "overlap": 10, "score": 0.78},
                        {"strategy": "Fixed-size", "chunk_size": 1024, "overlap": 20, "score": 0.75},
                        {"strategy": "Semantic", "boundary": "sentence", "min_size": 200, "score": 0.82},
                        {"strategy": "Hybrid", "base_size": 800, "semantic_boundaries": True, "score": 0.85}
                    ],
                    "best_configuration": {
                        "strategy": "Hybrid chunking",
                        "parameters": {"base_size": 800, "semantic_boundaries": True, "overlap": 15},
                        "improvement": "+13% context preservation, +8% retrieval accuracy"
                    }
                },
                
                "embedding_optimization": {
                    "tested_models": [
                        {"model": "all-mpnet-base-v2", "dimensions": 768, "mteb_score": 0.69, "domain_score": 0.72},
                        {"model": "e5-large-v2", "dimensions": 1024, "mteb_score": 0.72, "domain_score": 0.78},
                        {"model": "gte-large", "dimensions": 1024, "mteb_score": 0.71, "domain_score": 0.76},
                        {"model": "financial-bert-embeddings", "dimensions": 768, "mteb_score": 0.65, "domain_score": 0.85}
                    ],
                    "best_configuration": {
                        "model": "financial-bert-embeddings",
                        "dimensions": 768,
                        "justification": "Highest domain-specific performance despite lower general MTEB score",
                        "improvement": "+18% domain relevance, +12% retrieval precision"
                    }
                },
                
                "dimensionality_optimization": {
                    "pca_analysis": {
                        "original_dims": 768,
                        "tested_reductions": [384, 512, 640],
                        "results": {
                            "384": {"information_retention": 0.87, "storage_reduction": "50%", "speed_improvement": "40%"},
                            "512": {"information_retention": 0.93, "storage_reduction": "33%", "speed_improvement": "25%"},
                            "640": {"information_retention": 0.97, "storage_reduction": "17%", "speed_improvement": "15%"}
                        },
                        "selected_dimensions": 512,
                        "rationale": "Optimal balance of performance and efficiency"
                    }
                }
            },
            
            "step_3_integration": {
                "vector_database_selection": {
                    "tested_databases": [
                        {"db": "FAISS", "index": "IVF-PQ", "build_time": "45min", "query_latency": "12ms", "accuracy": 0.94},
                        {"db": "Qdrant", "index": "HNSW", "build_time": "65min", "query_latency": "8ms", "accuracy": 0.97},
                        {"db": "Milvus", "index": "IVF-FLAT", "build_time": "35min", "query_latency": "15ms", "accuracy": 0.95}
                    ],
                    "selected_configuration": {
                        "database": "Qdrant",
                        "index": "HNSW",
                        "parameters": {"M": 16, "efConstruction": 200, "efSearch": 100},
                        "justification": "Best accuracy-latency trade-off for financial domain"
                    }
                }
            }
        },
        
        "optimized_parameters": {
            "final_configuration": {
                "chunking": {
                    "strategy": "Hybrid chunking",
                    "base_size": 800,
                    "semantic_boundaries": True,
                    "overlap_percentage": 15,
                    "min_chunk_size": 200,
                    "normalization": ["whitespace", "unicode"]
                },
                "embedding": {
                    "model": "financial-bert-embeddings",
                    "dimensions": 512,
                    "batch_size": 32,
                    "pooling": "mean"
                },
                "vector_database": {
                    "database": "Qdrant",
                    "index_type": "HNSW",
                    "index_parameters": {"M": 16, "efConstruction": 200},
                    "storage_compression": "scalar_quantization"
                }
            },
            
            "performance_predictions": {
                "ingestion_metrics": {
                    "processing_speed": "85 documents/minute",
                    "total_indexing_time": "2.5 hours for 10,000 docs",
                    "storage_requirements": "850MB total",
                    "memory_usage": "4GB peak during indexing"
                },
                "quality_metrics": {
                    "context_preservation": 0.85,
                    "semantic_coherence": 0.88,
                    "information_completeness": 0.92
                },
                "cost_analysis": {
                    "monthly_storage_cost": "$45",
                    "compute_cost": "$120/month",
                    "total_operational_cost": "$165/month"
                }
            }
        }
    }
}

retrieval_evaluation_example = {
    "rag_retrieval_evaluation": {
        "scenario": {
            "indexed_documents": "10,000 financial documents",
            "query_patterns": "Regulatory compliance questions",
            "user_expectations": "Accurate, fast, comprehensive answers",
            "evaluation_dataset": "500 test queries with ground truth"
        },
        
        "baseline_configuration": {
            "retrieval_method": "Cosine similarity",
            "top_k": 5,
            "llm": "GPT-3.5-turbo",
            "prompt_template": "Basic Q&A template",
            "performance": {
                "precision_at_5": 0.68,
                "recall_at_5": 0.72,
                "answer_relevancy": 0.71,
                "faithfulness": 0.74,
                "response_time": "850ms"
            }
        },
        
        "optimization_process": {
            "retrieval_mechanism_optimization": {
                "tested_approaches": [
                    {
                        "method": "Pure vector similarity",
                        "parameters": {"similarity": "cosine", "top_k": 5},
                        "metrics": {"precision@5": 0.68, "recall@5": 0.72, "latency": "120ms"}
                    },
                    {
                        "method": "Hybrid search (vector + BM25)",
                        "parameters": {"vector_weight": 0.7, "bm25_weight": 0.3, "top_k": 10},
                        "metrics": {"precision@5": 0.75, "recall@5": 0.79, "latency": "180ms"}
                    },
                    {
                        "method": "Multi-stage retrieval",
                        "parameters": {"initial_k": 20, "rerank_k": 5, "reranker": "cross-encoder"},
                        "metrics": {"precision@5": 0.81, "recall@5": 0.76, "latency": "320ms"}
                    },
                    {
                        "method": "Contextual retrieval with query expansion",
                        "parameters": {"expansion_terms": 3, "context_window": 2, "top_k": 8},
                        "metrics": {"precision@5": 0.78, "recall@5": 0.83, "latency": "250ms"}
                    }
                ],
                "selected_approach": {
                    "method": "Multi-stage retrieval",
                    "justification": "Highest precision while maintaining acceptable latency",
                    "improvement": "+19% precision, +5% recall"
                }
            },
            
            "llm_optimization": {
                "tested_models": [
                    {
                        "model": "GPT-3.5-turbo",
                        "cost_per_1k_tokens": "$0.002",
                        "metrics": {"faithfulness": 0.74, "answer_relevancy": 0.71, "latency": "450ms"}
                    },
                    {
                        "model": "GPT-4",
                        "cost_per_1k_tokens": "$0.03",
                        "metrics": {"faithfulness": 0.89, "answer_relevancy": 0.85, "latency": "1200ms"}
                    },
                    {
                        "model": "Claude-3-Haiku",
                        "cost_per_1k_tokens": "$0.00025",
                        "metrics": {"faithfulness": 0.78, "answer_relevancy": 0.76, "latency": "380ms"}
                    },
                    {
                        "model": "Llama-2-70B (self-hosted)",
                        "cost_per_1k_tokens": "$0.0008",
                        "metrics": {"faithfulness": 0.81, "answer_relevancy": 0.79, "latency": "680ms"}
                    }
                ],
                "selected_model": {
                    "model": "Llama-2-70B (self-hosted)",
                    "justification": "Best cost-performance ratio with good quality",
                    "improvement": "+9% faithfulness, +11% answer relevancy"
                }
            },
            
            "prompt_optimization": {
                "tested_templates": [
                    {
                        "template": "Basic Q&A",
                        "structure": "Context: {context}\nQuestion: {question}\nAnswer:",
                        "metrics": {"answer_relevancy": 0.71, "completeness": 0.68}
                    },
                    {
                        "template": "Context-aware with instructions",
                        "structure": "You are a financial compliance expert. Use the following context to answer the question accurately...",
                        "metrics": {"answer_relevancy": 0.79, "completeness": 0.74}
                    },
                    {
                        "template": "Chain-of-thought reasoning",
                        "structure": "Think step by step about this financial question using the provided context...",
                        "metrics": {"answer_relevancy": 0.83, "completeness": 0.81}
                    },
                    {
                        "template": "Few-shot with examples",
                        "structure": "Here are examples of similar questions and answers... Now answer:",
                        "metrics": {"answer_relevancy": 0.85, "completeness": 0.79}
                    }
                ],
                "selected_template": {
                    "template": "Few-shot with examples",
                    "improvement": "+20% answer relevancy, +16% completeness"
                }
            }
        },
        
        "optimized_parameters": {
            "final_configuration": {
                "retrieval": {
                    "method": "Multi-stage retrieval",
                    "initial_retrieval_k": 20,
                    "final_k": 5,
                    "reranker": "cross-encoder-ms-marco-MiniLM",
                    "similarity_threshold": 0.7
                },
                "llm": {
                    "model": "Llama-2-70B",
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "context_window": 4096
                },
                "prompt": {
                    "template": "Few-shot with domain examples",
                    "examples_count": 3,
                    "instruction_emphasis": "financial accuracy"
                }
            },
            
            "performance_predictions": {
                "retrieval_metrics": {
                    "precision_at_5": 0.81,
                    "recall_at_5": 0.76,
                    "mrr": 0.87,
                    "ndcg_at_5": 0.83
                },
                "generation_metrics": {
                    "faithfulness": 0.81,
                    "answer_relevancy": 0.85,
                    "completeness": 0.79,
                    "rouge_l": 0.73
                },
                "system_metrics": {
                    "end_to_end_latency": "1.2 seconds",
                    "cost_per_query": "$0.012",
                    "throughput": "50 queries/minute"
                }
            }
        },
        
        "rouge_bleu_calculations": {
            "sample_evaluation": {
                "query": "What are the reporting requirements for derivative transactions under Dodd-Frank?",
                "ground_truth": "Under Dodd-Frank, derivative transactions must be reported to swap data repositories within specific timeframes, with cleared swaps reported by the end of the business day and uncleared swaps reported within 24 hours.",
                "generated_answer": "Dodd-Frank requires derivative transactions to be reported to registered swap data repositories. Cleared swaps must be reported by end of business day, while uncleared swaps have a 24-hour reporting window.",
                
                "rouge_scores": {
                    "rouge_1": {"precision": 0.85, "recall": 0.82, "f1": 0.835},
                    "rouge_2": {"precision": 0.78, "recall": 0.73, "f1": 0.755},
                    "rouge_l": {"precision": 0.81, "recall": 0.79, "f1": 0.80}
                },
                
                "bleu_score": {
                    "bleu_1": 0.82,
                    "bleu_2": 0.75,
                    "bleu_3": 0.68,
                    "bleu_4": 0.62,
                    "cumulative_bleu": 0.72
                }
            }
        }
    }
}

# Save the evaluation examples
with open('ingestion_evaluation_example.json', 'w') as f:
    json.dump(ingestion_evaluation_example, f, indent=2)

with open('retrieval_evaluation_example.json', 'w') as f:
    json.dump(retrieval_evaluation_example, f, indent=2)

print("RAG Evaluation Examples created successfully!")
print("\nIngestion Evaluation Summary:")
print(f"- Processing speed improvement: 70% (50→85 docs/min)")
print(f"- Context preservation improvement: 13% (0.75→0.85)")
print(f"- Storage efficiency: 33% reduction with minimal quality loss")

print("\nRetrieval Evaluation Summary:")
print(f"- Precision@5 improvement: 19% (0.68→0.81)")
print(f"- Answer relevancy improvement: 20% (0.71→0.85)")
print(f"- Cost optimization: 60% reduction per query while improving quality")