"""
Test script for the Advanced Research Orchestration system
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from advanced_orchestration import AdvancedResearchOrchestrator


async def test_advanced_orchestration():
    """Test the advanced research orchestration system"""
    try:
        print("🚀 Testing Advanced Research Orchestration")
        print("=" * 50)
        
        # Initialize components
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        orchestrator = AdvancedResearchOrchestrator(embedding_system, config)
        
        # Add test documents
        test_docs = [
            "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning approaches.",
            "Supervised learning uses labeled training data to learn patterns and make predictions on new data.",
            "Unsupervised learning discovers hidden patterns in data without using labeled examples.",
            "Reinforcement learning trains agents through interaction with an environment using rewards and penalties.",
            "Deep learning uses neural networks with multiple layers to automatically learn complex representations from data."
        ]
        
        print("📚 Adding test documents...")
        result = await embedding_system.add_documents(test_docs)
        print(f"Documents added: {result['status']}")
        
        # Test advanced orchestration
        query = "What are the different types of machine learning and how do they work?"
        
        print(f"\\n🔍 Running advanced orchestration for query: {query}")
        
        # Define a simple human feedback callback for testing
        async def mock_human_feedback(state):
            return "Continue with current approach. Quality looks good."
        
        # Execute orchestration
        result = await orchestrator.orchestrate_research(
            query=query,
            human_feedback_callback=mock_human_feedback
        )
        
        print("\\n📊 Results:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            final_result = result['result']
            print(f"Total findings: {final_result.get('total_findings', 0)}")
            print(f"Quality metrics: {final_result.get('quality_metrics', {})}")
            print(f"Completion status: {final_result.get('completion_status', 'unknown')}")
            
            # Display some findings
            findings = final_result.get('findings', [])
            if findings:
                print(f"\\n📋 Sample findings:")
                for i, finding in enumerate(findings[:3]):  # Show first 3
                    if isinstance(finding, dict):
                        content = finding.get('content', 'No content')[:100]
                        print(f"  {i+1}. {content}...")
                    else:
                        print(f"  {i+1}. {str(finding)[:100]}...")
            
            print(f"\\n💬 Process messages:")
            for msg in result.get('messages', [])[-3:]:  # Show last 3 messages
                print(f"  - {msg}")
        
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("\\n" + "=" * 50)
        print("✅ Advanced orchestration test completed!")
        
        return result['status'] == 'success'
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_advanced_orchestration())