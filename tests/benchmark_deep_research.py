"""
Deep Research Bench - Evaluation framework for research agent performance.
Based on patterns from open_deep_research and deepagents documentation.
"""

import asyncio
import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from research_workflow import ResearchWorkflowOrchestrator
from research_agent import DeepResearchAgent


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    query: str
    execution_time: float
    confidence_score: float
    findings_count: int
    sources_used: int
    iterations: int
    success: bool
    error_message: str = ""
    quality_metrics: Dict[str, float] = None


@dataclass 
class BenchmarkSuite:
    """Complete benchmark suite results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    average_execution_time: float
    average_confidence_score: float
    average_findings_count: float
    performance_metrics: Dict[str, Any]
    individual_results: List[BenchmarkResult]


class DeepResearchBench:
    """
    Comprehensive evaluation framework for Deep Research Agent.
    Implements evaluation patterns based on latest research agent benchmarking.
    """
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.benchmark_queries = self._load_benchmark_queries()
        self.results_history = []
    
    def _load_benchmark_queries(self) -> List[Dict[str, Any]]:
        """Load benchmark queries for different complexity levels"""
        return [
            {
                "query": "What are the main applications of machine learning in healthcare?",
                "category": "factual",
                "complexity": "simple",
                "expected_findings": 3,
                "expected_confidence": 0.8
            },
            {
                "query": "How do different neural network architectures compare for natural language processing tasks?", 
                "category": "comparative",
                "complexity": "medium",
                "expected_findings": 5,
                "expected_confidence": 0.7
            },
            {
                "query": "What are the emerging trends and future challenges in quantum computing for cryptography applications?",
                "category": "predictive",
                "complexity": "complex",
                "expected_findings": 4,
                "expected_confidence": 0.6
            },
            {
                "query": "Analyze the effectiveness of different data preprocessing techniques for time series forecasting models.",
                "category": "analytical", 
                "complexity": "complex",
                "expected_findings": 6,
                "expected_confidence": 0.75
            },
            {
                "query": "What is Python programming language used for?",
                "category": "factual",
                "complexity": "simple", 
                "expected_findings": 2,
                "expected_confidence": 0.9
            },
            {
                "query": "Compare supervised vs unsupervised learning approaches for customer segmentation in e-commerce.",
                "category": "comparative",
                "complexity": "medium",
                "expected_findings": 4,
                "expected_confidence": 0.7
            },
            {
                "query": "Evaluate the impact of transformer models on the evolution of natural language understanding systems.",
                "category": "evaluative",
                "complexity": "complex", 
                "expected_findings": 5,
                "expected_confidence": 0.65
            },
            {
                "query": "What are the key differences between relational and NoSQL databases?",
                "category": "comparative",
                "complexity": "medium",
                "expected_findings": 3,
                "expected_confidence": 0.8
            }
        ]
    
    async def _setup_test_environment(self) -> Tuple[AdvancedEmbeddingSystem, ResearchWorkflowOrchestrator]:
        """Setup test environment with sample data"""
        embedding_system = AdvancedEmbeddingSystem(self.config)
        orchestrator = ResearchWorkflowOrchestrator(embedding_system, self.config)
        
        # Load comprehensive test dataset
        test_documents = [
            # Machine Learning & AI
            "Machine learning is transforming healthcare through diagnostic imaging, drug discovery, and personalized treatment plans.",
            "Supervised learning algorithms like Random Forest and SVM are widely used for medical diagnosis applications.",
            "Deep learning models, particularly CNNs, have shown remarkable success in medical image analysis and pathology detection.",
            "Natural language processing helps extract insights from electronic health records and medical literature.",
            
            # Neural Networks & NLP
            "Transformer models like BERT and GPT have revolutionized natural language processing tasks.",
            "Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are effective for sequence modeling.",
            "Convolutional Neural Networks (CNNs) excel at computer vision tasks and image recognition.",
            "Attention mechanisms allow models to focus on relevant parts of input sequences for better performance.",
            
            # Quantum Computing
            "Quantum computing promises to break current cryptographic methods through Shor's algorithm implementation.",
            "Quantum key distribution provides theoretically unbreakable encryption methods for secure communication.",
            "Current quantum computers face challenges with decoherence and error rates in practical applications.",
            "Post-quantum cryptography is being developed to resist attacks from quantum computers.",
            
            # Data Science & Analytics
            "Time series preprocessing includes techniques like normalization, differencing, and seasonal decomposition.",
            "Feature engineering and selection are crucial for improving machine learning model performance.",
            "Cross-validation and hyperparameter tuning help prevent overfitting in predictive models.",
            "Data cleaning and outlier detection are essential steps in the data science pipeline.",
            
            # Programming & Databases
            "Python is a versatile programming language used for web development, data science, automation, and AI.",
            "Python's extensive library ecosystem includes NumPy, Pandas, Scikit-learn, and TensorFlow for data science.",
            "Relational databases use structured schemas and SQL for complex queries and ACID transactions.",
            "NoSQL databases like MongoDB and Cassandra offer flexible schemas and horizontal scalability.",
            
            # Customer Analytics
            "Supervised learning methods like clustering and classification help segment customers based on behavior patterns.",
            "Unsupervised learning techniques like K-means clustering reveal hidden customer segments without labeled data.",
            "Customer lifetime value prediction uses regression models to forecast long-term customer profitability.",
            "Recommendation systems use collaborative filtering and content-based approaches for personalization."
        ]
        
        # Add documents to the system
        result = await embedding_system.add_documents(test_documents)
        if result["status"] != "success":
            raise Exception(f"Failed to setup test environment: {result}")
        
        return embedding_system, orchestrator
    
    async def run_single_benchmark(self, query_data: Dict[str, Any], orchestrator: ResearchWorkflowOrchestrator) -> BenchmarkResult:
        """Run a single benchmark test"""
        query = query_data["query"]
        start_time = time.time()
        
        try:
            # Execute research
            result = await orchestrator.orchestrate_research(query)
            execution_time = time.time() - start_time
            
            if result["status"] == "success":
                research_result = result["result"]
                artifacts = result.get("artifacts", {})
                
                return BenchmarkResult(
                    query=query,
                    execution_time=execution_time,
                    confidence_score=research_result.confidence_score if research_result else 0.0,
                    findings_count=len(research_result.findings) if research_result else 0,
                    sources_used=len(research_result.sources_used) if research_result else 0,
                    iterations=artifacts.get("iteration_count", 0),
                    success=True,
                    quality_metrics=artifacts.get("quality_assessment", {})
                )
            else:
                return BenchmarkResult(
                    query=query,
                    execution_time=execution_time,
                    confidence_score=0.0,
                    findings_count=0,
                    sources_used=0,
                    iterations=0,
                    success=False,
                    error_message=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                query=query,
                execution_time=execution_time,
                confidence_score=0.0,
                findings_count=0,
                sources_used=0,
                iterations=0,
                success=False,
                error_message=str(e)
            )
    
    async def run_benchmark_suite(self, max_concurrent: int = 2) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        print("🚀 Starting Deep Research Bench evaluation...")
        
        # Setup test environment
        embedding_system, orchestrator = await self._setup_test_environment()
        
        # Run benchmarks with controlled concurrency
        results = []
        for i in range(0, len(self.benchmark_queries), max_concurrent):
            batch = self.benchmark_queries[i:i+max_concurrent]
            
            print(f"📊 Running benchmark batch {i//max_concurrent + 1}/{(len(self.benchmark_queries) + max_concurrent - 1)//max_concurrent}")
            
            # Execute batch concurrently
            batch_tasks = [self.run_single_benchmark(query_data, orchestrator) for query_data in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(BenchmarkResult(
                        query="Exception occurred",
                        execution_time=0.0,
                        confidence_score=0.0,
                        findings_count=0,
                        sources_used=0,
                        iterations=0,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    results.append(result)
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.success]
        
        total_tests = len(results)
        successful_tests = len(successful_results)
        failed_tests = total_tests - successful_tests
        
        if successful_results:
            avg_execution_time = statistics.mean([r.execution_time for r in successful_results])
            avg_confidence = statistics.mean([r.confidence_score for r in successful_results])
            avg_findings = statistics.mean([r.findings_count for r in successful_results])
        else:
            avg_execution_time = 0.0
            avg_confidence = 0.0
            avg_findings = 0.0
        
        # Performance metrics
        performance_metrics = {
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "median_execution_time": statistics.median([r.execution_time for r in successful_results]) if successful_results else 0,
            "execution_time_std": statistics.stdev([r.execution_time for r in successful_results]) if len(successful_results) > 1 else 0,
            "confidence_score_std": statistics.stdev([r.confidence_score for r in successful_results]) if len(successful_results) > 1 else 0,
            "total_findings": sum([r.findings_count for r in successful_results]),
            "total_sources": sum([r.sources_used for r in successful_results]),
            "total_iterations": sum([r.iterations for r in successful_results])
        }
        
        return BenchmarkSuite(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            average_execution_time=avg_execution_time,
            average_confidence_score=avg_confidence,
            average_findings_count=avg_findings,
            performance_metrics=performance_metrics,
            individual_results=results
        )
    
    def generate_benchmark_report(self, suite: BenchmarkSuite, output_path: str = None) -> str:
        """Generate comprehensive benchmark report"""
        if output_path is None:
            output_path = f"benchmark_report_{int(time.time())}.json"
        
        # Create detailed report
        report = {
            "benchmark_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": suite.total_tests,
                "successful_tests": suite.successful_tests,
                "failed_tests": suite.failed_tests,
                "success_rate": f"{suite.performance_metrics['success_rate']:.2%}",
                "average_execution_time": f"{suite.average_execution_time:.2f}s",
                "average_confidence_score": f"{suite.average_confidence_score:.3f}",
                "average_findings_count": f"{suite.average_findings_count:.1f}"
            },
            "performance_metrics": suite.performance_metrics,
            "detailed_results": [asdict(result) for result in suite.individual_results],
            "analysis": self._generate_analysis(suite)
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
    
    def _generate_analysis(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Generate analysis insights from benchmark results"""
        successful_results = [r for r in suite.individual_results if r.success]
        
        analysis = {
            "performance_analysis": {
                "fastest_query": min(successful_results, key=lambda x: x.execution_time).query if successful_results else "N/A",
                "slowest_query": max(successful_results, key=lambda x: x.execution_time).query if successful_results else "N/A",
                "highest_confidence": max(successful_results, key=lambda x: x.confidence_score).query if successful_results else "N/A",
                "most_findings": max(successful_results, key=lambda x: x.findings_count).query if successful_results else "N/A"
            },
            "quality_insights": {
                "queries_above_threshold": len([r for r in successful_results if r.confidence_score > 0.7]),
                "queries_with_multiple_sources": len([r for r in successful_results if r.sources_used > 1]),
                "average_iterations_per_query": statistics.mean([r.iterations for r in successful_results]) if successful_results else 0
            },
            "failure_analysis": {
                "failed_queries": [r.query for r in suite.individual_results if not r.success],
                "common_errors": list(set([r.error_message for r in suite.individual_results if not r.success and r.error_message]))
            }
        }
        
        return analysis
    
    def print_benchmark_summary(self, suite: BenchmarkSuite):
        """Print formatted benchmark summary"""
        print("\n" + "="*80)
        print("🏆 DEEP RESEARCH BENCH - EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {suite.total_tests}")
        print(f"   Successful: {suite.successful_tests} ({suite.performance_metrics['success_rate']:.1%})")
        print(f"   Failed: {suite.failed_tests}")
        
        print(f"\n⏱️  EXECUTION METRICS:")
        print(f"   Average Time: {suite.average_execution_time:.2f}s")
        print(f"   Median Time: {suite.performance_metrics['median_execution_time']:.2f}s")
        print(f"   Time Std Dev: {suite.performance_metrics['execution_time_std']:.2f}s")
        
        print(f"\n🎯 QUALITY METRICS:")
        print(f"   Average Confidence: {suite.average_confidence_score:.3f}")
        print(f"   Average Findings: {suite.average_findings_count:.1f}")
        print(f"   Total Sources Used: {suite.performance_metrics['total_sources']}")
        print(f"   Total Iterations: {suite.performance_metrics['total_iterations']}")
        
        # Show individual results
        print(f"\n📊 INDIVIDUAL RESULTS:")
        for i, result in enumerate(suite.individual_results, 1):
            status = "✅" if result.success else "❌"
            print(f"   {i:2d}. {status} {result.query[:60]}{'...' if len(result.query) > 60 else ''}")
            if result.success:
                print(f"       Time: {result.execution_time:.2f}s | Confidence: {result.confidence_score:.3f} | Findings: {result.findings_count}")
            else:
                print(f"       Error: {result.error_message}")
        
        print("\n" + "="*80)


async def main():
    """Run the Deep Research Bench evaluation"""
    # Initialize benchmark
    config = ResearchConfig()
    bench = DeepResearchBench(config)
    
    # Run benchmark suite
    suite = await bench.run_benchmark_suite(max_concurrent=1)  # Sequential for stability
    
    # Print results
    bench.print_benchmark_summary(suite)
    
    # Generate report
    report_path = bench.generate_benchmark_report(suite)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return suite


if __name__ == "__main__":
    # Run the benchmark
    results = asyncio.run(main())