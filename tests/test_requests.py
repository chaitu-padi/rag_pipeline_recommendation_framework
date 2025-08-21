"""
Test requests for the RAG Pipeline Recommendation API
"""
import json
import httpx
import pytest
from typing import Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "http://127.0.0.1:8000/api/v1"
TEST_DATA_DIR = Path(__file__).parent

# Sample test requests for different scenarios
TEST_REQUESTS = {
    "enterprise_legal": {
        "description": "Enterprise legal document processing with high accuracy requirements",
        "request": {
            "primary_data_type": "pdf",
            "secondary_data_types": ["word", "text"],
            "document_complexity": "highly_complex",
            "content_domain": "legal",
            "current_volume": "large",
            "average_document_length": "long",
            "update_frequency": "daily",
            "language_requirements": ["english"],
            "expected_growth_rate": "20% yearly",
            "metadata_importance": "critical",
            
            "primary_use_case": "semantic_search",
            "expected_query_complexity": "analytical",
            "accuracy_tolerance": "zero_tolerance",
            "latency_requirement": "interactive",
            "preferred_response_type": "citation_heavy",
            "citation_requirements": "mandatory",
            "expected_queries_per_day": "1000-10000",
            "concurrent_users": "10-50",
            
            "budget_range": "high",
            "team_expertise": "advanced",
            "data_sensitivity_level": "confidential",
            "industry_domain": "legal",
            
            "scalability_requirements": "high",
            "monitoring_depth": "comprehensive"
        }
    },
    
    "startup_support": {
        "description": "Cost-efficient customer support system for a startup",
        "request": {
            "primary_data_type": "text",
            "secondary_data_types": ["html"],
            "document_complexity": "moderate",
            "content_domain": "customer_support",
            "current_volume": "small",
            "average_document_length": "short",
            "update_frequency": "real_time",
            "language_requirements": ["english", "spanish"],
            "expected_growth_rate": "30% monthly",
            "metadata_importance": "medium",
            
            "primary_use_case": "question_answering",
            "expected_query_complexity": "simple_factual",
            "accuracy_tolerance": "moderate_tolerance",
            "latency_requirement": "real_time",
            "preferred_response_type": "brief_answers",
            "citation_requirements": "helpful",
            "expected_queries_per_day": "100-1000",
            "concurrent_users": "1-10",
            
            "budget_range": "minimal",
            "team_expertise": "basic_technical",
            "data_sensitivity_level": "internal",
            "industry_domain": "ecommerce",
            
            "scalability_requirements": "moderate",
            "monitoring_depth": "standard"
        }
    },
    
    "research_academic": {
        "description": "Academic research papers analysis system",
        "request": {
            "primary_data_type": "pdf",
            "secondary_data_types": [],
            "document_complexity": "complex",
            "content_domain": "academic",
            "current_volume": "medium",
            "average_document_length": "long",
            "update_frequency": "weekly",
            "language_requirements": ["english"],
            "expected_growth_rate": "10% monthly",
            "metadata_importance": "high",
            
            "primary_use_case": "research_assistance",
            "expected_query_complexity": "analytical",
            "accuracy_tolerance": "low_tolerance",
            "latency_requirement": "responsive",
            "preferred_response_type": "detailed_explanations",
            "citation_requirements": "required",
            "expected_queries_per_day": "100-1000",
            "concurrent_users": "10-50",
            
            "budget_range": "moderate",
            "team_expertise": "intermediate",
            "data_sensitivity_level": "internal",
            "industry_domain": "research",
            
            "scalability_requirements": "moderate",
            "monitoring_depth": "standard"
        }
    },
    
    "technical_docs": {
        "description": "Technical documentation and code analysis system",
        "request": {
            "primary_data_type": "code",
            "secondary_data_types": ["markdown", "text"],
            "document_complexity": "complex",
            "content_domain": "technical",
            "current_volume": "medium",
            "average_document_length": "medium",
            "update_frequency": "hourly",
            "language_requirements": ["english"],
            "expected_growth_rate": "15% monthly",
            "metadata_importance": "high",
            
            "primary_use_case": "semantic_search",
            "expected_query_complexity": "procedural",
            "accuracy_tolerance": "low_tolerance",
            "latency_requirement": "interactive",
            "preferred_response_type": "structured_summaries",
            "citation_requirements": "required",
            "expected_queries_per_day": "1000-10000",
            "concurrent_users": "50-200",
            
            "budget_range": "moderate",
            "team_expertise": "expert",
            "data_sensitivity_level": "internal",
            "industry_domain": "software",
            
            "scalability_requirements": "high",
            "monitoring_depth": "comprehensive"
        }
    }
}

import sys
import json
import logging
from pathlib import Path
import pytest
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path(__file__).parent

class TestRecommendationAPI:
    """Test suite for the RAG Pipeline Recommendation API"""
    
    @pytest.fixture
    def test_client(self):
        """Create a test client"""
        return httpx.AsyncClient(base_url=BASE_URL)
    
    @pytest.fixture
    def test_data(self):
        """Load test data"""
        try:
            test_data_path = TEST_DATA_DIR / "test_requests.json"
            if not test_data_path.exists():
                with open(test_data_path, "w") as f:
                    json.dump(TEST_REQUESTS, f, indent=2)
                logger.info(f"Created test data file at {test_data_path}")
            
            with open(test_data_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return TEST_REQUESTS

    @pytest.mark.asyncio
    async def test_all_scenarios(self, test_client, test_data):
        """Test all scenarios and validate responses"""
        for scenario, data in test_data.items():
            logger.info(f"\nTesting scenario: {scenario}")
            logger.info(f"Description: {data['description']}")
            
            try:
                response = await test_client.post(
                    "/recommend-comprehensive",
                    json=data["request"]
                )
                
                # Validate response
                assert response.status_code == 200, f"Failed with status {response.status_code}"
                result = response.json()
                
                # Validate structure
                assert "rag_pipelines" in result, "Missing rag_pipelines in response"
                assert len(result["rag_pipelines"]) > 0, "No pipeline recommendations returned"
                
                # Validate first pipeline
                pipeline = result["rag_pipelines"][0]
                required_fields = ["name", "description", "when_to_use", "trade_offs"]
                for field in required_fields:
                    assert field in pipeline, f"Missing {field} in pipeline response"
                
                logger.info(f"✓ {scenario} test passed")
                logger.info(f"  Received {len(result['rag_pipelines'])} recommendations")
                
            except Exception as e:
                logger.error(f"Error in {scenario}: {str(e)}")
                raise

    @pytest.mark.asyncio
    async def test_error_handling(self, test_client):
        """Test API error handling"""
        invalid_request = {"invalid": "data"}
        
        response = await test_client.post(
            "/recommend-comprehensive",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422], "Should return client error for invalid data"
        error_data = response.json()
        assert "detail" in error_data, "Error response should include detail"

def main():
    """Main function for manual testing"""
    import asyncio
    import sys
    
    async def run_tests():
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            test_suite = TestRecommendationAPI()
            await test_suite.test_all_scenarios(client, TEST_REQUESTS)
    
    try:
        asyncio.run(run_tests())
        logger.info("All tests completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Example curl commands for manual testing:
"""
# Enterprise Legal System
curl -X POST http://127.0.0.1:8000/api/v1/recommend-comprehensive \
  -H "Content-Type: application/json" \
  -d @test_requests.json --data-binary "@-" | jq '.rag_pipelines[0]'

# Startup Support System
curl -X POST http://127.0.0.1:8000/api/v1/recommend-comprehensive \
  -H "Content-Type: application/json" \
  -d @test_requests.json --data-binary "@-" | jq '.rag_pipelines[0]'
"""
