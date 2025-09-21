"""
Test configuration and utilities for the Deep Research Agent test suite.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Test configuration constants
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_DB_DIR = None
MOCK_RESPONSES_FILE = Path(__file__).parent / "mock_responses.json"

# Mock responses for consistent testing
MOCK_LLM_RESPONSES = {
    "query_decomposition": {
        "simple": '{"subqueries": ["What is the topic?"], "keywords": ["main keyword"], "research_goals": ["understand basics"], "complexity": "simple", "required_sources": 2}',
        "medium": '{"subqueries": ["What is X?", "How does X work?"], "keywords": ["keyword1", "keyword2"], "research_goals": ["understand concept", "analyze application"], "complexity": "medium", "required_sources": 3}',
        "complex": '{"subqueries": ["What is X?", "How does X compare to Y?", "What are future trends?"], "keywords": ["keyword1", "keyword2", "keyword3"], "research_goals": ["understand concept", "comparative analysis", "trend prediction"], "complexity": "complex", "required_sources": 5}'
    },
    "research_planning": {
        "basic": '{"research_objective": "Understand the topic", "strategy": {"strategy_name": "comprehensive", "search_methods": ["similarity"], "validation_level": "standard", "iteration_limit": 2, "confidence_threshold": 0.7}, "workflow_steps": [{"step_name": "search", "step_type": "search", "parameters": {}, "dependencies": [], "success_criteria": ["found relevant documents"]}], "quality_gates": ["accuracy"], "estimated_duration": 10}',
        "advanced": '{"research_objective": "Comprehensive analysis", "strategy": {"strategy_name": "deep_analysis", "search_methods": ["similarity", "keyword"], "validation_level": "high", "iteration_limit": 3, "confidence_threshold": 0.8}, "workflow_steps": [{"step_name": "search", "step_type": "search", "parameters": {}, "dependencies": [], "success_criteria": ["found relevant documents"]}, {"step_name": "analyze", "step_type": "analyze", "parameters": {}, "dependencies": ["search"], "success_criteria": ["analysis complete"]}], "quality_gates": ["accuracy", "completeness"], "estimated_duration": 20}'
    },
    "quality_assessment": {
        "good": '{"completeness_score": 0.8, "accuracy_score": 0.85, "relevance_score": 0.9, "source_diversity": 0.7, "confidence_level": 0.8, "areas_for_improvement": ["more diverse sources"], "validation_status": "good"}',
        "excellent": '{"completeness_score": 0.95, "accuracy_score": 0.9, "relevance_score": 0.95, "source_diversity": 0.85, "confidence_level": 0.9, "areas_for_improvement": [], "validation_status": "excellent"}',
        "poor": '{"completeness_score": 0.5, "accuracy_score": 0.6, "relevance_score": 0.7, "source_diversity": 0.4, "confidence_level": 0.5, "areas_for_improvement": ["more sources needed", "improve accuracy"], "validation_status": "needs improvement"}'
    },
    "synthesis": {
        "basic": "Based on the research findings, the topic can be understood through the following key points: main concept, applications, and benefits.",
        "detailed": "The comprehensive analysis reveals multiple facets of the topic. Key findings include: 1) Core concepts and definitions, 2) Practical applications and use cases, 3) Comparative advantages and limitations, 4) Future trends and implications.",
        "comparative": "Comparative analysis shows that approach A differs from approach B in several key ways: methodology, performance, scalability, and use cases. Each has distinct advantages depending on the specific requirements."
    }
}

# Test document collections for different domains
TEST_DOCUMENT_COLLECTIONS = {
    "machine_learning": [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
        "Supervised learning algorithms learn from labeled training data to make predictions on new data.",
        "Unsupervised learning finds hidden patterns in data without using labeled examples.",
        "Reinforcement learning trains agents to make decisions through interaction with an environment.",
        "Deep learning uses neural networks with multiple layers to learn complex representations.",
        "Feature engineering is the process of selecting and transforming variables for machine learning models.",
        "Cross-validation helps assess how well a model will generalize to independent datasets.",
        "Overfitting occurs when a model learns noise in training data rather than generalizable patterns."
    ],
    "web_development": [
        "HTML provides the structure and content of web pages using markup elements.",
        "CSS controls the visual presentation and layout of HTML elements on web pages.",
        "JavaScript adds interactivity and dynamic behavior to web applications.",
        "React is a JavaScript library for building user interfaces with reusable components.",
        "Node.js enables server-side JavaScript execution for backend development.",
        "REST APIs provide a standardized way for applications to communicate over HTTP.",
        "Database integration allows web applications to store and retrieve persistent data.",
        "Responsive design ensures web applications work well across different device sizes."
    ],
    "data_science": [
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Python and R are popular programming languages for data analysis and statistical computing.",
        "Data cleaning and preprocessing typically consume 80% of a data scientist's time.",
        "Exploratory data analysis helps understand data patterns before building models.",
        "Statistical hypothesis testing helps validate assumptions and draw conclusions from data.",
        "Data visualization communicates findings effectively to stakeholders and decision makers.",
        "Big data technologies handle datasets too large for traditional processing methods.",
        "Ethical considerations in data science include privacy, bias, and algorithmic fairness."
    ]
}

def setup_test_environment():
    """Setup global test environment"""
    global TEMP_DB_DIR
    
    # Create temporary directory for test databases
    TEMP_DB_DIR = tempfile.mkdtemp(prefix="deep_research_test_")
    
    # Create test data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ["RESEARCH_AGENT_TEST_MODE"] = "true"
    os.environ["RESEARCH_AGENT_LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    return TEMP_DB_DIR

def cleanup_test_environment():
    """Cleanup test environment"""
    global TEMP_DB_DIR
    
    if TEMP_DB_DIR and os.path.exists(TEMP_DB_DIR):
        shutil.rmtree(TEMP_DB_DIR)
        TEMP_DB_DIR = None
    
    # Clean up environment variables
    os.environ.pop("RESEARCH_AGENT_TEST_MODE", None)
    os.environ.pop("RESEARCH_AGENT_LOG_LEVEL", None)

def get_test_config(collection_name: str = "test_collection") -> Dict[str, Any]:
    """Get test configuration with temporary database"""
    if not TEMP_DB_DIR:
        setup_test_environment()
    
    return {
        "database_config": {
            "collection_name": collection_name,
            "persist_directory": TEMP_DB_DIR,
            "embedding_model": "mxbai-embed-large",
            "distance_metric": "cosine"
        },
        "llm_config": {
            "model": "llama3.2",
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": 30
        },
        "retrieval_config": {
            "search_type": "similarity",
            "k": 5,
            "score_threshold": 0.3,
            "max_marginal_relevance": True
        },
        "research_config": {
            "max_iterations": 2,  # Reduced for faster testing
            "confidence_threshold": 0.7,
            "quality_threshold": 0.6,
            "timeout_seconds": 60  # Reduced for testing
        }
    }

def create_test_documents(domain: str = "machine_learning") -> list:
    """Get test documents for a specific domain"""
    return TEST_DOCUMENT_COLLECTIONS.get(domain, TEST_DOCUMENT_COLLECTIONS["machine_learning"])

def get_mock_response(category: str, quality: str = "basic") -> str:
    """Get mock LLM response for testing"""
    return MOCK_LLM_RESPONSES.get(category, {}).get(quality, "Mock response")

# Test utilities for assertions
def assert_research_result_valid(result):
    """Assert that a research result has valid structure"""
    assert hasattr(result, 'query')
    assert hasattr(result, 'findings')
    assert hasattr(result, 'synthesis')
    assert hasattr(result, 'confidence_score')
    assert 0 <= result.confidence_score <= 1
    assert isinstance(result.findings, list)

def assert_quality_assessment_valid(assessment):
    """Assert that a quality assessment has valid structure"""
    assert hasattr(assessment, 'completeness_score')
    assert hasattr(assessment, 'accuracy_score')
    assert hasattr(assessment, 'relevance_score')
    assert hasattr(assessment, 'confidence_level')
    
    # All scores should be between 0 and 1
    assert 0 <= assessment.completeness_score <= 1
    assert 0 <= assessment.accuracy_score <= 1
    assert 0 <= assessment.relevance_score <= 1
    assert 0 <= assessment.confidence_level <= 1

def assert_finding_valid(finding):
    """Assert that a research finding has valid structure"""
    assert hasattr(finding, 'content')
    assert hasattr(finding, 'source')
    assert hasattr(finding, 'relevance_score')
    assert hasattr(finding, 'confidence_score')
    assert len(finding.content) > 0
    assert 0 <= finding.relevance_score <= 1
    assert 0 <= finding.confidence_score <= 1

# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.description} completed in {self.duration:.3f} seconds")

# Import time for PerformanceTimer
import time