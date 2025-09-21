"""
Test runner for the Deep Research Agent system.
Provides easy interface to run different types of tests.
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from test_config import setup_test_environment, cleanup_test_environment
from benchmark_deep_research import DeepResearchBench


def run_unit_tests(verbose: bool = False):
    """Run unit tests using pytest"""
    try:
        import pytest
        
        test_args = [str(Path(__file__).parent / "test_deep_research_agent.py")]
        
        if verbose:
            test_args.extend(["-v", "--tb=short"])
        else:
            test_args.extend(["-q"])
        
        print("🧪 Running unit tests...")
        setup_test_environment()
        
        exit_code = pytest.main(test_args)
        
        cleanup_test_environment()
        
        if exit_code == 0:
            print("✅ All unit tests passed!")
        else:
            print("❌ Some unit tests failed.")
        
        return exit_code == 0
        
    except ImportError:
        print("❌ pytest is required to run unit tests. Install with: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        cleanup_test_environment()
        return False


async def run_benchmark_tests(max_concurrent: int = 1):
    """Run benchmark tests"""
    try:
        print("🏆 Running benchmark tests...")
        setup_test_environment()
        
        from config import ResearchConfig
        config = ResearchConfig()
        bench = DeepResearchBench(config)
        
        # Run benchmark suite
        suite = await bench.run_benchmark_suite(max_concurrent=max_concurrent)
        
        # Print results
        bench.print_benchmark_summary(suite)
        
        # Generate report
        report_path = bench.generate_benchmark_report(suite)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        cleanup_test_environment()
        
        # Consider benchmark successful if success rate > 70%
        success_rate = suite.performance_metrics['success_rate']
        if success_rate > 0.7:
            print(f"✅ Benchmark completed successfully! Success rate: {success_rate:.1%}")
            return True
        else:
            print(f"⚠️  Benchmark completed with low success rate: {success_rate:.1%}")
            return False
            
    except Exception as e:
        print(f"❌ Error running benchmark tests: {e}")
        cleanup_test_environment()
        return False


def run_integration_test():
    """Run a quick integration test to verify the system works"""
    try:
        print("🔧 Running integration test...")
        setup_test_environment()
        
        # Import required components
        from config import ResearchConfig
        from embedding_system import AdvancedEmbeddingSystem
        from research_workflow import ResearchWorkflowOrchestrator
        
        async def integration_test():
            # Setup components
            config = ResearchConfig()
            embedding_system = AdvancedEmbeddingSystem(config)
            orchestrator = ResearchWorkflowOrchestrator(embedding_system, config)
            
            # Add test documents
            test_docs = [
                "Python is a high-level programming language known for its simplicity and readability.",
                "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based.",
                "Data science involves extracting insights from large datasets using statistical methods."
            ]
            
            result = await embedding_system.add_documents(test_docs)
            if result["status"] != "success":
                raise Exception(f"Failed to add documents: {result}")
            
            # Test search functionality
            search_results = await embedding_system.similarity_search("programming languages", k=2)
            if len(search_results) == 0:
                raise Exception("Search returned no results")
            
            print(f"✅ Integration test passed! Found {len(search_results)} relevant documents.")
            return True
        
        # Run the async integration test
        success = asyncio.run(integration_test())
        cleanup_test_environment()
        return success
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        cleanup_test_environment()
        return False


def run_quick_smoke_test():
    """Run a quick smoke test to verify imports work"""
    try:
        print("💨 Running smoke test...")
        
        # Test imports
        from config import ResearchConfig
        from embedding_system import AdvancedEmbeddingSystem
        from research_agent import DeepResearchAgent
        from research_workflow import ResearchWorkflowOrchestrator
        from quality_control import QualityController
        from report_generator import ReportGenerator
        
        # Test basic initialization
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        agent = DeepResearchAgent(embedding_system, config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, config)
        quality_controller = QualityController(config)  # Only takes config
        report_gen = ReportGenerator(config)
        
        print("✅ Smoke test passed! All components can be imported and initialized.")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Deep Research Agent Test Runner")
    parser.add_argument(
        "test_type", 
        choices=["smoke", "integration", "unit", "benchmark", "all"],
        help="Type of test to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=1,
        help="Max concurrent tests for benchmark (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Deep Research Agent Test Runner")
    print("=" * 50)
    
    success = True
    
    if args.test_type == "smoke" or args.test_type == "all":
        success &= run_quick_smoke_test()
        print()
    
    if args.test_type == "integration" or args.test_type == "all":
        success &= run_integration_test()
        print()
    
    if args.test_type == "unit" or args.test_type == "all":
        success &= run_unit_tests(verbose=args.verbose)
        print()
    
    if args.test_type == "benchmark" or args.test_type == "all":
        success &= asyncio.run(run_benchmark_tests(max_concurrent=args.concurrent))
        print()
    
    print("=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())