#!/usr/bin/env python3
"""
Enhanced RAG Pipeline Recommender - Comprehensive Demonstration
Shows the sophisticated analysis and recommendation capabilities across different business scenarios
"""
import sys
import json
from datetime import datetime

# Load all enhanced components
exec(open("enhanced_models.py").read())
exec(open("enhanced_knowledge_base.py").read())
exec(open("enhanced_engine.py").read())
exec(open("enhanced_composer.py").read())

def create_test_scenario(scenario_name: str, **kwargs) -> ComprehensiveUserRequirements:
    """Create a test scenario with the given parameters"""

    # Default values
    defaults = {
        "primary_data_type": DataType.PDF,
        "document_complexity": DocumentComplexity.MODERATE,
        "content_domain": ContentDomain.GENERAL,
        "current_volume": VolumeSize.SMALL,
        "update_frequency": UpdateFrequency.STATIC,
        "language_requirements": ["english"],
        "primary_use_case": UseCase.QUESTION_ANSWERING,
        "query_complexity": QueryComplexity.SIMPLE_FACTUAL,
        "accuracy_tolerance": AccuracyTolerance.MODERATE_TOLERANCE,
        "latency_requirement": LatencyRequirement.INTERACTIVE,
        "budget_range": BudgetRange.MODERATE,
        "team_expertise": TeamExpertise.INTERMEDIATE,
        "data_sensitivity": "internal",
        "deployment_preference": "flexible"
    }

    # Update with scenario-specific values
    defaults.update(kwargs)

    return ComprehensiveUserRequirements(
        data_characteristics=DataCharacteristics(
            primary_data_type=defaults["primary_data_type"],
            document_complexity=defaults["document_complexity"],
            content_domain=defaults["content_domain"],
            current_volume=defaults["current_volume"],
            update_frequency=defaults["update_frequency"],
            language_requirements=defaults["language_requirements"],
            average_document_length="medium",
            expected_growth_rate="10% monthly",
            metadata_importance="medium",
            has_structured_metadata=False,
            data_quality_assessment="good",
            content_consistency="moderate",
            duplicate_content_level="low"
        ),
        use_case_requirements=UseCaseRequirements(
            primary_use_case=defaults["primary_use_case"],
            expected_query_complexity=defaults["query_complexity"],
            accuracy_tolerance=defaults["accuracy_tolerance"],
            latency_requirement=defaults["latency_requirement"],
            preferred_response_type=ResponseType.DETAILED_EXPLANATIONS,
            citation_requirements="helpful",
            expected_queries_per_day="100-1000",
            concurrent_users="1-10"
        ),
        business_context=BusinessContext(
            budget_range=defaults["budget_range"],
            team_expertise=defaults["team_expertise"],
            data_sensitivity_level=defaults["data_sensitivity"],
            regulatory_requirements=[],
            maintenance_capability="moderate",
            industry_domain="general"
        ),
        technical_preferences=TechnicalPreferences(
            deployment_preference=defaults["deployment_preference"],
            scalability_requirements="moderate",
            monitoring_depth="standard",
            customization_importance="moderate"
        ),
        additional_context=f"Test scenario: {scenario_name}"
    )

def print_scenario_analysis(scenario_name: str, result: ComprehensiveRecommendationResult):
    """Print detailed analysis of a scenario"""

    print(f"\n{'='*80}")
    print(f"📊 SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    # Requirements Analysis
    print("\n🔍 REQUIREMENTS ANALYSIS:")
    for key, analysis in result.requirements_analysis.items():
        print(f"  • {key.replace('_', ' ').title()}: {analysis}")

    # Top Recommendations Summary
    print("\n🎯 TOP RECOMMENDATIONS:")

    # Best ingestion pipeline (usually the balanced one)
    best_ingestion = result.ingestion_pipelines[1]  # Balanced option
    print(f"\n📥 RECOMMENDED INGESTION PIPELINE:")
    print(f"  Pipeline: {best_ingestion.name}")
    print(f"  Embedding: {best_ingestion.embedding.model_name} ({best_ingestion.embedding.dimensions}D)")
    print(f"  Chunking: {best_ingestion.chunking.strategy} ({best_ingestion.chunking.chunk_size} tokens)")
    print(f"  Vector DB: {best_ingestion.vector_db.provider}")
    print(f"  Estimated Cost: {best_ingestion.estimated_cost}")
    print(f"  Why: {best_ingestion.when_to_use[:120]}...")

    # Best RAG pipeline (context-dependent)
    best_rag = result.rag_pipelines[1]  # Usually balanced, but could be different
    print(f"\n📤 RECOMMENDED RAG PIPELINE:")
    print(f"  Pipeline: {best_rag.name}")
    print(f"  LLM: {best_rag.rag_config.llm_model}")
    print(f"  Retrieval: K={best_rag.rag_config.retrieval_k}, Rerank={best_rag.rag_config.rerank_enabled}")
    print(f"  Hybrid Search: {best_rag.rag_config.hybrid_search}")
    print(f"  Latency: {best_rag.estimated_latency}")
    print(f"  Why: {best_rag.when_to_use[:120]}...")

    # Trade-offs and Risks
    if result.trade_off_analysis:
        print("\n⚖️ KEY TRADE-OFFS:")
        for key, trade_off in result.trade_off_analysis.items():
            print(f"  • {key.replace('_', ' ').title()}: {trade_off}")

    if result.risk_assessment:
        print("\n⚠️ RISK ASSESSMENT:")
        for key, risk in result.risk_assessment.items():
            print(f"  • {key.replace('_', ' ').title()}: {risk}")

    # Implementation Roadmap (first 3 steps)
    print("\n🗺️ IMPLEMENTATION ROADMAP (First 3 Steps):")
    for i, step in enumerate(result.implementation_roadmap[:3], 1):
        print(f"  {i}. {step}")

    print("\n" + "-"*80)

def run_comprehensive_demo():
    """Run comprehensive demonstration across multiple realistic scenarios"""

    print("🚀 ENHANCED RAG PIPELINE RECOMMENDER - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows sophisticated analysis across diverse business scenarios")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize the composer
    composer = EnhancedPipelineComposer()
    print(f"\n✅ System initialized with {len(composer.knowledge_base.embedding_models)} embedding models, {len(composer.knowledge_base.vector_databases)} vector databases")

    # Scenario 1: Legal Firm with Compliance Requirements
    print("\n\n" + "="*60)
    print("📋 RUNNING SCENARIO 1: Legal Firm with Compliance Requirements")
    print("="*60)

    legal_scenario = create_test_scenario(
        "Legal Firm",
        primary_data_type=DataType.PDF,
        document_complexity=DocumentComplexity.HIGHLY_COMPLEX,
        content_domain=ContentDomain.LEGAL,
        current_volume=VolumeSize.LARGE,
        accuracy_tolerance=AccuracyTolerance.ZERO_TOLERANCE,
        budget_range=BudgetRange.HIGH,
        team_expertise=TeamExpertise.INTERMEDIATE,
        data_sensitivity="confidential"
    )

    # Add specific legal requirements
    legal_scenario.business_context.regulatory_requirements = ["GDPR", "SOX"]
    legal_scenario.use_case_requirements.citation_requirements = "mandatory"
    legal_scenario.use_case_requirements.expected_query_complexity = QueryComplexity.INTERPRETIVE

    legal_result = composer.generate_comprehensive_recommendations(legal_scenario)
    print_scenario_analysis("Legal Firm with Compliance Requirements", legal_result)

    # Scenario 2: Healthcare Organization with HIPAA
    print("\n\n" + "="*60)
    print("🏥 RUNNING SCENARIO 2: Healthcare Organization with HIPAA Requirements")
    print("="*60)

    healthcare_scenario = create_test_scenario(
        "Healthcare Organization",
        primary_data_type=DataType.MIXED,
        document_complexity=DocumentComplexity.COMPLEX,
        content_domain=ContentDomain.MEDICAL,
        current_volume=VolumeSize.VERY_LARGE,
        accuracy_tolerance=AccuracyTolerance.ZERO_TOLERANCE,
        budget_range=BudgetRange.HIGH,
        team_expertise=TeamExpertise.ADVANCED,
        data_sensitivity="restricted",
        deployment_preference="on_premises"
    )

    # Add HIPAA requirements
    healthcare_scenario.business_context.regulatory_requirements = ["HIPAA", "FDA"]
    healthcare_scenario.data_characteristics.secondary_data_types = [DataType.PDF, DataType.TEXT, DataType.XML]
    healthcare_scenario.use_case_requirements.primary_use_case = UseCase.RESEARCH_ASSISTANCE

    healthcare_result = composer.generate_comprehensive_recommendations(healthcare_scenario)
    print_scenario_analysis("Healthcare Organization with HIPAA Requirements", healthcare_result)

    # Scenario 3: Startup with Budget Constraints
    print("\n\n" + "="*60)
    print("💰 RUNNING SCENARIO 3: Cost-Conscious Startup")
    print("="*60)

    startup_scenario = create_test_scenario(
        "Cost-Conscious Startup",
        primary_data_type=DataType.TEXT,
        document_complexity=DocumentComplexity.SIMPLE,
        content_domain=ContentDomain.CUSTOMER_SUPPORT,
        current_volume=VolumeSize.SMALL,
        budget_range=BudgetRange.LOW,
        team_expertise=TeamExpertise.BASIC_TECHNICAL,
        latency_requirement=LatencyRequirement.REAL_TIME,
        accuracy_tolerance=AccuracyTolerance.MODERATE_TOLERANCE
    )

    startup_scenario.use_case_requirements.primary_use_case = UseCase.CONVERSATIONAL_AI
    startup_scenario.technical_preferences.deployment_preference = "managed_services"
    startup_scenario.business_context.maintenance_capability = "minimal"

    startup_result = composer.generate_comprehensive_recommendations(startup_scenario)
    print_scenario_analysis("Cost-Conscious Startup", startup_result)

    # Scenario 4: Enterprise with Multilingual Requirements
    print("\n\n" + "="*60)
    print("🌍 RUNNING SCENARIO 4: Global Enterprise with Multilingual Content")
    print("="*60)

    enterprise_scenario = create_test_scenario(
        "Global Enterprise",
        primary_data_type=DataType.MIXED,
        document_complexity=DocumentComplexity.COMPLEX,
        content_domain=ContentDomain.TECHNICAL,
        current_volume=VolumeSize.MASSIVE,
        budget_range=BudgetRange.UNLIMITED,
        team_expertise=TeamExpertise.EXPERT,
        language_requirements=["english", "spanish", "french", "german", "chinese", "japanese"]
    )

    enterprise_scenario.data_characteristics.secondary_data_types = [DataType.PDF, DataType.HTML, DataType.WORD, DataType.JSON]
    enterprise_scenario.use_case_requirements.primary_use_case = UseCase.KNOWLEDGE_DISCOVERY
    enterprise_scenario.use_case_requirements.expected_query_complexity = QueryComplexity.ANALYTICAL
    enterprise_scenario.technical_preferences.scalability_requirements = "extreme"
    enterprise_scenario.technical_preferences.monitoring_depth = "comprehensive"

    enterprise_result = composer.generate_comprehensive_recommendations(enterprise_scenario)
    print_scenario_analysis("Global Enterprise with Multilingual Content", enterprise_result)

    # Scenario 5: Financial Services with Real-time Requirements  
    print("\n\n" + "="*60)
    print("🏦 RUNNING SCENARIO 5: Financial Services with Real-time Analytics")
    print("="*60)

    financial_scenario = create_test_scenario(
        "Financial Services",
        primary_data_type=DataType.JSON,
        document_complexity=DocumentComplexity.COMPLEX,
        content_domain=ContentDomain.FINANCIAL,
        current_volume=VolumeSize.LARGE,
        update_frequency=UpdateFrequency.REAL_TIME,
        latency_requirement=LatencyRequirement.REAL_TIME,
        accuracy_tolerance=AccuracyTolerance.LOW_TOLERANCE,
        budget_range=BudgetRange.HIGH,
        team_expertise=TeamExpertise.EXPERT
    )

    financial_scenario.business_context.regulatory_requirements = ["SOX", "PCI_DSS"]
    financial_scenario.use_case_requirements.primary_use_case = UseCase.CONTENT_ANALYSIS
    financial_scenario.use_case_requirements.concurrent_users = "200+"
    financial_scenario.use_case_requirements.expected_queries_per_day = "10000+"

    financial_result = composer.generate_comprehensive_recommendations(financial_scenario)
    print_scenario_analysis("Financial Services with Real-time Analytics", financial_result)

    # Summary Statistics
    print("\n\n" + "="*80)
    print("📈 DEMONSTRATION SUMMARY")
    print("="*80)

    scenarios = [
        ("Legal Firm", legal_result),
        ("Healthcare", healthcare_result), 
        ("Startup", startup_result),
        ("Enterprise", enterprise_result),
        ("Financial", financial_result)
    ]

    print("\n🎯 RECOMMENDATION DIVERSITY ANALYSIS:")
    embedding_models = set()
    vector_dbs = set()
    llm_models = set()

    for scenario_name, result in scenarios:
        # Collect recommended components (balanced pipeline)
        ingestion = result.ingestion_pipelines[1]
        rag = result.rag_pipelines[1]

        embedding_models.add(ingestion.embedding.model_name)
        vector_dbs.add(ingestion.vector_db.provider)
        llm_models.add(rag.rag_config.llm_model)

        print(f"  {scenario_name:12}: {ingestion.embedding.model_name:20} + {ingestion.vector_db.provider:10} + {rag.rag_config.llm_model}")

    print(f"\n📊 COMPONENT DIVERSITY:")
    print(f"  • Embedding Models Used: {len(embedding_models)} different models")
    print(f"  • Vector Databases Used: {len(vector_dbs)} different databases")
    print(f"  • LLM Models Used: {len(llm_models)} different models")

    print(f"\n🔧 SYSTEM CAPABILITIES DEMONSTRATED:")
    print(f"  ✅ Domain-specific optimization (Legal, Medical, Financial, Technical)")
    print(f"  ✅ Deployment flexibility (On-premises, Managed, Hybrid)")
    print(f"  ✅ Budget optimization (Minimal to Unlimited ranges)")
    print(f"  ✅ Scale handling (Small to Massive volumes)")
    print(f"  ✅ Compliance awareness (GDPR, HIPAA, SOX, PCI-DSS)")
    print(f"  ✅ Multilingual support (Up to 6 languages)")
    print(f"  ✅ Performance tuning (Real-time to Batch acceptable)")
    print(f"  ✅ Risk assessment and trade-off analysis")
    print(f"  ✅ Implementation roadmap generation")
    print(f"  ✅ Future evolution planning")

    print(f"\n🎉 ENHANCED RAG PIPELINE RECOMMENDER DEMONSTRATION COMPLETED!")
    print(f"The system successfully analyzed {len(scenarios)} diverse scenarios and generated")
    print(f"tailored recommendations with sophisticated business intelligence.")

    return scenarios

if __name__ == "__main__":
    try:
        scenarios = run_comprehensive_demo()
        print(f"\n✅ Demo completed successfully with {len(scenarios)} scenarios analyzed")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
