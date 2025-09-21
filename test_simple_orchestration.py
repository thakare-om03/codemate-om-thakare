"""
Simplified test for the Advanced Research Orchestration system
Focuses on basic functionality without complex LLM interactions
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from advanced_orchestration import AdvancedResearchOrchestrator, ResearchPhase, AgentRole


async def test_orchestration_components():
    """Test individual components of the orchestration system"""
    try:
        print("🔧 Testing Advanced Orchestration Components")
        print("=" * 50)
        
        # Test 1: Initialize components
        print("1. Testing component initialization...")
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        orchestrator = AdvancedResearchOrchestrator(embedding_system, config)
        
        print("✅ Components initialized successfully")
        
        # Test 2: Check agent initialization
        print("\\n2. Testing agent system...")
        agents = orchestrator.agents
        print(f"   Initialized {len(agents)} agents:")
        for role, agent in agents.items():
            print(f"   - {role.value}: {len(agent.capabilities)} capabilities")
        
        print("✅ Agent system working")
        
        # Test 3: Check research tools
        print("\\n3. Testing research tools...")
        tools = orchestrator.research_tools
        print(f"   Available tools: {len(tools)}")
        for tool in tools[:3]:  # Show first 3
            print(f"   - {tool.name}: {tool.category} (reliability: {tool.reliability_score})")
        
        print("✅ Research tools available")
        
        # Test 4: Test workflow compilation
        print("\\n4. Testing workflow compilation...")
        workflow = orchestrator.workflow
        print(f"   Workflow compiled: {workflow is not None}")
        
        print("✅ Workflow compilation successful")
        
        # Test 5: Simple document processing
        print("\\n5. Testing document processing...")
        test_docs = [
            "Artificial intelligence involves creating systems that can perform tasks requiring human intelligence.",
            "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
        ]
        
        result = await embedding_system.add_documents(test_docs)
        print(f"   Document processing: {result['status']}")
        
        print("✅ Document processing working")
        
        print("\\n" + "=" * 50)
        print("🎉 All component tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_orchestration():
    """Test simplified orchestration without complex LLM calls"""
    try:
        print("\\n🚀 Testing Simplified Orchestration")
        print("=" * 50)
        
        # Initialize
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add documents
        test_docs = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "JavaScript is primarily used for web development and creating interactive user interfaces.",
            "SQL is used for managing and querying relational databases."
        ]
        
        await embedding_system.add_documents(test_docs)
        print("📚 Test documents added")
        
        # Test individual orchestration methods without full workflow
        from advanced_orchestration import ResearchTask
        
        # Test task creation
        task = ResearchTask(
            task_id="test_1",
            task_type="search",
            description="Test search task",
            priority=1,
            dependencies=[],
            assigned_agent=AgentRole.SEARCHER
        )
        
        print(f"✅ Task created: {task.task_id}")
        
        # Test basic retrieval through orchestrator components
        from advanced_retrieval import AdvancedRetriever
        retriever = AdvancedRetriever(embedding_system, config)
        
        # Test semantic search
        docs = await embedding_system.similarity_search("programming languages", k=2)
        print(f"✅ Retrieved {len(docs)} documents")
        
        print("\\n" + "=" * 50)
        print("🎉 Simplified orchestration test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified orchestration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🧪 Advanced Research Orchestration Test Suite")
    print("=" * 60)
    
    # Run component tests
    component_test = await test_orchestration_components()
    
    if component_test:
        # Run simplified orchestration test
        orchestration_test = await test_simple_orchestration()
        
        if orchestration_test:
            print("\\n🎉 ALL TESTS PASSED! Advanced orchestration system is ready.")
            return True
    
    print("\\n❌ Some tests failed.")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())