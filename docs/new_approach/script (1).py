# Create evaluation metrics and calculation framework
evaluation_metrics = {
    "evaluation_metrics": {
        "retrieval_metrics": {
            "precision_at_k": {
                "formula": "Precision@k = (Relevant documents retrieved in top-k) / k",
                "description": "Measures the proportion of retrieved documents that are relevant",
                "calculation": "For each query, count relevant docs in top-k results, divide by k, average across queries",
                "interpretation": "Higher values indicate better precision",
                "typical_range": "0.0 to 1.0",
                "use_case": "When precision is more important than recall"
            },
            "recall_at_k": {
                "formula": "Recall@k = (Relevant documents retrieved in top-k) / (Total relevant documents)",
                "description": "Measures the proportion of relevant documents that are retrieved",
                "calculation": "For each query, count relevant docs in top-k, divide by total relevant docs, average",
                "interpretation": "Higher values indicate better coverage of relevant documents",
                "typical_range": "0.0 to 1.0",
                "use_case": "When comprehensive retrieval is critical"
            },
            "mean_reciprocal_rank": {
                "formula": "MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is position of first relevant document",
                "description": "Measures how quickly relevant documents are found",
                "calculation": "For each query, find rank of first relevant doc, take reciprocal, average across queries",
                "interpretation": "Higher values indicate relevant documents appear earlier in results",
                "typical_range": "0.0 to 1.0",
                "use_case": "When finding the first relevant result quickly is important"
            },
            "ndcg_at_k": {
                "formula": "NDCG@k = DCG@k / IDCG@k",
                "description": "Normalized discounted cumulative gain, accounts for relevance levels and position",
                "calculation": "Calculate DCG with relevance weights and position discounting, normalize by ideal DCG",
                "interpretation": "Higher values indicate better ranking quality with position consideration",
                "typical_range": "0.0 to 1.0",
                "use_case": "When both relevance and ranking order matter"
            },
            "context_precision": {
                "formula": "Context Precision = (Σ Precision@k * v_k) / Σ v_k",
                "description": "RAGAS metric measuring precision of retrieved contexts",
                "calculation": "Uses LLM to evaluate if retrieved contexts are useful for answering the question",
                "interpretation": "Higher scores indicate more relevant retrieved contexts",
                "typical_range": "0.0 to 1.0",
                "use_case": "RAG-specific evaluation of context relevance"
            },
            "context_recall": {
                "formula": "Context Recall = |Relevant sentences in retrieved context| / |Total relevant sentences|",
                "description": "RAGAS metric measuring recall of retrieved contexts",
                "calculation": "Compares retrieved context with ground truth to measure coverage",
                "interpretation": "Higher scores indicate better coverage of relevant information",
                "typical_range": "0.0 to 1.0",
                "use_case": "Ensuring comprehensive information retrieval"
            }
        },
        
        "generation_metrics": {
            "bleu_score": {
                "formula": "BLEU = BP * exp(Σ w_n * log(p_n))",
                "description": "Measures n-gram overlap between generated and reference text",
                "calculation": "Calculate precision for 1-4 grams, apply brevity penalty, geometric mean",
                "interpretation": "Higher scores indicate better similarity to reference text",
                "typical_range": "0.0 to 1.0",
                "use_case": "When exact phrase matching is important",
                "limitations": "May not capture semantic similarity well"
            },
            "rouge_scores": {
                "rouge_1": {
                    "formula": "ROUGE-1 = Overlapping unigrams / Total unigrams in reference",
                    "description": "Measures unigram overlap",
                    "use_case": "Basic content overlap assessment"
                },
                "rouge_2": {
                    "formula": "ROUGE-2 = Overlapping bigrams / Total bigrams in reference", 
                    "description": "Measures bigram overlap",
                    "use_case": "Phrase-level content similarity"
                },
                "rouge_l": {
                    "formula": "ROUGE-L = F-measure based on longest common subsequence",
                    "description": "Measures longest common subsequence",
                    "use_case": "Structural similarity assessment"
                }
            },
            "faithfulness": {
                "formula": "Faithfulness = |Supported claims| / |Total claims|",
                "description": "RAGAS metric measuring factual consistency with retrieved context",
                "calculation": "Extract claims from answer, verify each against retrieved context using LLM",
                "interpretation": "Higher scores indicate better grounding in provided context",
                "typical_range": "0.0 to 1.0",
                "use_case": "Preventing hallucinations in RAG responses"
            },
            "answer_relevancy": {
                "formula": "Answer Relevancy = cosine_similarity(original_question, generated_questions)",
                "description": "RAGAS metric measuring how well answer addresses the question",
                "calculation": "Generate questions from answer, measure similarity to original question",
                "interpretation": "Higher scores indicate better question-answer alignment",
                "typical_range": "0.0 to 1.0",
                "use_case": "Ensuring answers directly address user queries"
            },
            "semantic_similarity": {
                "formula": "Semantic Similarity = cosine_similarity(embedding(generated), embedding(reference))",
                "description": "Measures semantic similarity using embeddings",
                "calculation": "Generate embeddings for generated and reference text, calculate cosine similarity",
                "interpretation": "Higher scores indicate better semantic alignment",
                "typical_range": "-1.0 to 1.0 (typically 0.0 to 1.0)",
                "use_case": "When semantic meaning matters more than exact wording"
            }
        },
        
        "system_metrics": {
            "latency_metrics": {
                "query_latency": {
                    "measurement": "Time from query submission to response delivery",
                    "units": "milliseconds",
                    "components": ["Embedding generation", "Vector search", "Context retrieval", "LLM inference"],
                    "targets": {
                        "real_time": "< 200ms",
                        "interactive": "< 1000ms", 
                        "batch": "< 5000ms"
                    }
                },
                "indexing_latency": {
                    "measurement": "Time to index new documents",
                    "units": "documents per second",
                    "factors": ["Document size", "Chunking complexity", "Embedding generation", "Index updates"]
                }
            },
            "throughput_metrics": {
                "queries_per_second": "Maximum concurrent queries handled",
                "documents_indexed_per_hour": "Indexing throughput rate",
                "concurrent_users": "Maximum simultaneous users supported"
            },
            "resource_metrics": {
                "memory_usage": "RAM consumption for vectors and models",
                "storage_requirements": "Disk space for indexes and embeddings",
                "cpu_utilization": "Processing power consumption",
                "gpu_usage": "GPU utilization for model inference"
            },
            "cost_metrics": {
                "cost_per_query": "Total cost divided by number of queries",
                "infrastructure_cost": "Monthly hosting and compute costs",
                "api_costs": "External API usage costs",
                "total_cost_of_ownership": "Annual operational costs including maintenance"
            }
        },
        
        "quality_metrics": {
            "hallucination_rate": {
                "measurement": "Percentage of responses containing unsupported claims",
                "calculation": "Manual review or automated fact-checking against source documents",
                "target": "< 5% for production systems"
            },
            "consistency_score": {
                "measurement": "Consistency of responses to similar queries",
                "calculation": "Measure response similarity for paraphrased questions",
                "target": "> 85% similarity for equivalent queries"
            },
            "completeness_score": {
                "measurement": "Coverage of expected information in responses",
                "calculation": "Compare response content against comprehensive reference answers",
                "target": "> 80% information coverage"
            }
        }
    },
    
    "iterative_optimization": {
        "methodology": {
            "baseline_establishment": {
                "steps": [
                    "1. Implement basic RAG pipeline with default parameters",
                    "2. Run evaluation on representative test set", 
                    "3. Record baseline metrics across all dimensions",
                    "4. Identify primary optimization targets"
                ],
                "baseline_parameters": {
                    "chunk_size": 1024,
                    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                    "vector_db": "FAISS with flat index",
                    "retrieval_k": 5,
                    "llm": "GPT-3.5-turbo"
                }
            },
            "optimization_cycles": {
                "cycle_structure": [
                    "1. Parameter hypothesis generation",
                    "2. Controlled parameter modification",
                    "3. Evaluation with consistent test set",
                    "4. Statistical significance testing",
                    "5. Parameter selection and iteration"
                ],
                "optimization_order": [
                    "1. Chunking strategy optimization",
                    "2. Embedding model selection",
                    "3. Vector database and indexing",
                    "4. Retrieval mechanism tuning",
                    "5. LLM selection and prompt optimization",
                    "6. End-to-end system tuning"
                ]
            },
            "multi_objective_optimization": {
                "pareto_optimization": "Find optimal trade-offs between competing objectives",
                "weighted_scoring": "Combine multiple metrics with user-defined weights",
                "constraint_satisfaction": "Meet minimum thresholds for critical metrics",
                "iterative_refinement": "Gradually improve multiple objectives simultaneously"
            }
        },
        
        "evaluation_framework": {
            "test_set_requirements": {
                "size": "Minimum 500 queries for statistical significance",
                "diversity": "Cover different query types and complexity levels",
                "ground_truth": "High-quality reference answers for each query",
                "difficulty_distribution": "Mix of easy, medium, and hard queries"
            },
            "cross_validation": {
                "k_fold": "5-fold cross-validation for robust evaluation",
                "temporal_splits": "Time-based splits for real-world data",
                "domain_splits": "Domain-specific evaluation sets"
            },
            "statistical_analysis": {
                "significance_testing": "t-tests for metric comparisons",
                "confidence_intervals": "95% confidence intervals for metrics",
                "effect_size": "Cohen's d for practical significance",
                "multiple_comparison_correction": "Bonferroni correction for multiple tests"
            }
        }
    }
}

# Save the evaluation metrics framework
with open('rag_evaluation_metrics.json', 'w') as f:
    json.dump(evaluation_metrics, f, indent=2)

print("RAG Evaluation Metrics framework created successfully!")
print(f"Retrieval metrics defined: {len(evaluation_metrics['evaluation_metrics']['retrieval_metrics'])}")
print(f"Generation metrics defined: {len(evaluation_metrics['evaluation_metrics']['generation_metrics'])}")
print(f"System metrics categories: {len(evaluation_metrics['evaluation_metrics']['system_metrics'])}")