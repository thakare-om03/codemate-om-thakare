"""
Comprehensive test suite for the Deep Research Agent system.
Based on latest testing patterns and Deep Research Bench evaluation framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core testing imports
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Deep Research Agent imports
from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from research_agent import DeepResearchAgent, ResearchFinding, ResearchResult
from research_workflow import ResearchWorkflowOrchestrator, ResearchPlan, QualityAssessment
from quality_control import QualityController


class TestConfig:
    """Test configuration and fixtures"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def research_config(self, temp_db_path):
        """Create test research configuration"""
        config = ResearchConfig()
        config.database_config["collection_name"] = "test_collection"
        config.database_config["persist_directory"] = temp_db_path
        return config
    
    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama model responses"""
        return {
            "content": "Test response from mock Ollama model",
            "additional_kwargs": {},
            "response_metadata": {"model": "llama3.2", "created_at": "2024-01-01"}
        }


class TestEmbeddingSystem(TestConfig):
    """Test the Advanced Embedding System"""
    
    @pytest.mark.asyncio
    async def test_embedding_system_initialization(self, research_config):
        """Test embedding system can be initialized correctly"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        assert embedding_system is not None
        assert embedding_system.config == research_config
    
    @pytest.mark.asyncio
    async def test_document_processing(self, research_config, temp_db_path):
        """Test document addition and processing"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        
        # Create test documents
        test_docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "Natural language processing helps computers understand human language."
        ]
        
        # Test document addition
        result = await embedding_system.add_documents(test_docs)
        assert result["status"] == "success"
        assert "documents_added" in result
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, research_config):
        """Test similarity search functionality"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        
        # Add test documents first
        test_docs = [
            "Python is a programming language used for data science.",
            "JavaScript is commonly used for web development.",
            "SQL is used for database queries and management."
        ]
        await embedding_system.add_documents(test_docs)
        
        # Test search
        query = "programming languages for data analysis"
        results = await embedding_system.similarity_search(query, k=2)
        
        assert len(results) <= 2
        assert all(hasattr(result, 'page_content') for result in results)


class TestResearchAgent(TestConfig):
    """Test the Deep Research Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, research_config):
        """Test research agent initialization"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        agent = DeepResearchAgent(embedding_system, research_config)
        
        assert agent is not None
        assert agent.config == research_config
        assert agent.embedding_system == embedding_system
    
    @pytest.mark.asyncio
    async def test_query_decomposition(self, research_config):
        """Test query decomposition functionality"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        agent = DeepResearchAgent(embedding_system, research_config)
        
        with patch.object(agent.llm, 'agenerate') as mock_generate:
            mock_generate.return_value = Mock(
                generations=[[Mock(text='{"subqueries": ["What is AI?", "How does ML work?"], "keywords": ["artificial intelligence", "machine learning"], "research_goals": ["understand AI basics"], "complexity": "medium", "required_sources": 3}')]]
            )
            
            query = "Explain artificial intelligence and machine learning"
            result = await agent._decompose_query(query)
            
            assert result is not None
            assert hasattr(result, 'subqueries')
            assert hasattr(result, 'keywords')
    
    @pytest.mark.asyncio
    async def test_research_execution(self, research_config):
        """Test end-to-end research execution"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        agent = DeepResearchAgent(embedding_system, research_config)
        
        # Add test documents
        test_docs = [
            "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Deep learning is part of a broader family of machine learning methods."
        ]
        await embedding_system.add_documents(test_docs)
        
        # Mock LLM responses
        with patch.object(agent.llm, 'agenerate') as mock_generate:
            # Mock decomposition response
            mock_generate.return_value = Mock(
                generations=[[Mock(text='{"subqueries": ["What is AI?"], "keywords": ["AI"], "research_goals": ["understand AI"], "complexity": "simple", "required_sources": 2}')]]
            )
            
            query = "What is artificial intelligence?"
            result = await agent.research(query)
            
            assert result is not None
            assert hasattr(result, 'query')
            assert hasattr(result, 'findings')


class TestResearchWorkflow(TestConfig):
    """Test the Research Workflow Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, research_config):
        """Test workflow orchestrator initialization"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        assert orchestrator is not None
        assert orchestrator.config == research_config
        assert orchestrator.embedding_system == embedding_system
        assert orchestrator.workflow_graph is not None
    
    @pytest.mark.asyncio
    async def test_research_planning(self, research_config):
        """Test research planning node"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        # Test planning with mock state
        test_state = {
            "messages": [],
            "original_query": "Test query about AI",
            "research_objective": "",
            "current_step_index": 0,
            "search_results": [],
            "findings": [],
            "intermediate_results": {},
            "validation_results": [],
            "confidence_scores": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "strategy_changes": [],
            "research_artifacts": {}
        }
        
        with patch.object(orchestrator.llm, 'agenerate') as mock_generate:
            mock_generate.return_value = Mock(
                generations=[[Mock(text='{"research_objective": "Test objective", "strategy": {"strategy_name": "comprehensive", "search_methods": ["similarity"], "validation_level": "standard", "iteration_limit": 3, "confidence_threshold": 0.7}, "workflow_steps": [], "quality_gates": ["accuracy"], "estimated_duration": 10}')]]
            )
            
            result = await orchestrator._planning_node(test_state)
            
            assert "research_plan" in result
            assert "messages" in result
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, research_config):
        """Test quality assessment functionality"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        # Create test findings
        test_findings = [
            ResearchFinding(
                content="Test finding about AI",
                source="test_source",
                relevance_score=0.8,
                confidence_score=0.9,
                citations=["citation1"]
            )
        ]
        
        with patch.object(orchestrator.llm, 'agenerate') as mock_generate:
            mock_generate.return_value = Mock(
                generations=[[Mock(text='{"completeness_score": 0.8, "accuracy_score": 0.9, "relevance_score": 0.85, "source_diversity": 0.7, "confidence_level": 0.8, "areas_for_improvement": ["more sources needed"], "validation_status": "good"}')]]
            )
            
            assessment = await orchestrator._assess_research_quality(test_findings, "test query")
            
            assert isinstance(assessment, QualityAssessment)
            assert assessment.completeness_score > 0
            assert assessment.accuracy_score > 0


class TestQualityControl(TestConfig):
    """Test the Quality Control System"""
    
    def test_quality_controller_initialization(self, research_config):
        """Test quality controller initialization"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        controller = QualityController(embedding_system, research_config)
        
        assert controller is not None
        assert controller.config == research_config
    
    @pytest.mark.asyncio
    async def test_finding_validation(self, research_config):
        """Test research finding validation"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        controller = QualityController(embedding_system, research_config)
        
        test_finding = ResearchFinding(
            content="Test finding about machine learning applications",
            source="test_source",
            relevance_score=0.8,
            confidence_score=0.9,
            citations=["test citation"]
        )
        
        is_valid = await controller.validate_finding(test_finding, "machine learning")
        assert isinstance(is_valid, bool)


class TestIntegration(TestConfig):
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, research_config):
        """Test complete research workflow from query to result"""
        # Initialize components
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        # Add test knowledge base
        test_docs = [
            "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based.",
            "Supervised learning requires labeled training data to learn patterns.",
            "Unsupervised learning finds hidden patterns in data without labels.",
            "Reinforcement learning learns through interaction with an environment."
        ]
        
        await embedding_system.add_documents(test_docs)
        
        # Mock all LLM calls for integration test
        with patch.object(orchestrator.llm, 'agenerate') as mock_generate:
            # Set up sequential mock responses for different workflow stages
            mock_responses = [
                # Planning response
                Mock(text='{"research_objective": "Understand ML types", "strategy": {"strategy_name": "comprehensive", "search_methods": ["similarity"], "validation_level": "standard", "iteration_limit": 2, "confidence_threshold": 0.7}, "workflow_steps": [{"step_name": "search", "step_type": "search", "parameters": {}, "dependencies": [], "success_criteria": ["found relevant documents"]}], "quality_gates": ["accuracy"], "estimated_duration": 15}'),
                # Search strategy response
                Mock(text='{"search_methods": ["similarity"], "search_parameters": {"k": 5}, "query_expansion": ["machine learning", "ML types"], "expected_sources": 3}'),
                # Analysis response
                Mock(text='Machine learning can be categorized into three main types: supervised, unsupervised, and reinforcement learning.'),
                # Quality assessment response
                Mock(text='{"completeness_score": 0.9, "accuracy_score": 0.95, "relevance_score": 0.9, "source_diversity": 0.8, "confidence_level": 0.9, "areas_for_improvement": [], "validation_status": "excellent"}'),
                # Synthesis response
                Mock(text='Based on the research, machine learning encompasses three primary approaches: supervised learning with labeled data, unsupervised learning for pattern discovery, and reinforcement learning through environmental interaction.')
            ]
            
            mock_generate.return_value = Mock(generations=[[resp] for resp in mock_responses])
            
            # Execute research
            result = await orchestrator.orchestrate_research(
                "What are the different types of machine learning algorithms?"
            )
            
            assert result["status"] == "success"
            assert result["result"] is not None
            assert "artifacts" in result
    
    @pytest.mark.asyncio 
    async def test_error_handling(self, research_config):
        """Test system error handling and recovery"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        # Test with invalid query that should trigger error handling
        with patch.object(orchestrator.llm, 'agenerate') as mock_generate:
            mock_generate.side_effect = Exception("Simulated LLM error")
            
            result = await orchestrator.orchestrate_research("test query")
            
            assert result["status"] == "error"
            assert "error" in result
            assert result["result"] is None


class TestPerformance(TestConfig):
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_research_requests(self, research_config):
        """Test system performance with concurrent requests"""
        embedding_system = AdvancedEmbeddingSystem(research_config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, research_config)
        
        # Add test documents
        test_docs = ["Test document " + str(i) for i in range(10)]
        await embedding_system.add_documents(test_docs)
        
        # Mock LLM for performance test
        with patch.object(orchestrator.llm, 'agenerate') as mock_generate:
            mock_generate.return_value = Mock(
                generations=[[Mock(text='{"research_objective": "test", "strategy": {"strategy_name": "fast", "search_methods": ["similarity"], "validation_level": "basic", "iteration_limit": 1, "confidence_threshold": 0.5}, "workflow_steps": [], "quality_gates": [], "estimated_duration": 5}')]]
            )
            
            # Create concurrent research tasks
            queries = [f"Test query {i}" for i in range(3)]
            tasks = [orchestrator.orchestrate_research(query) for query in queries]
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all succeeded
            assert len(results) == 3
            for result in results:
                assert not isinstance(result, Exception)
                assert result["status"] == "success"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])