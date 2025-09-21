"""
Deep Research Agent - Main Application
Comprehensive research agent with local LLM, vector database, and multi-step reasoning
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_research_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Local imports
from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from research_workflow import ResearchWorkflowOrchestrator
from report_generator import ReportGenerator
from quality_control import QualityController
from interactive_interface import InteractiveResearchInterface, CLIInterface


class DeepResearchAgent:
    """
    Main Deep Research Agent application
    
    Features:
    - Local embedding generation with Ollama
    - Multi-step reasoning and query decomposition
    - ChromaDB vector storage
    - Multiple document format support
    - Interactive web and CLI interfaces
    - Quality control and evaluation
    - Multi-format report generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Deep Research Agent"""
        self.config = ResearchConfig()
        self.config.create_directories()
        
        # Initialize core components
        self.embedding_system = None
        self.orchestrator = None
        self.report_generator = None
        self.quality_controller = None
        
        logger.info("Deep Research Agent initialized")
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize embedding system
            self.embedding_system = AdvancedEmbeddingSystem(self.config)
            logger.info("* Embedding system initialized")
            
            # Initialize orchestrator
            self.orchestrator = ResearchWorkflowOrchestrator(self.embedding_system, self.config)
            logger.info("* Research workflow orchestrator initialized")
            
            # Initialize report generator
            self.report_generator = ReportGenerator(self.config)
            logger.info("* Report generator initialized")
            
            # Initialize quality controller
            self.quality_controller = QualityController(self.config)
            logger.info("* Quality controller initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def conduct_research(self, 
                             query: str,
                             research_depth: str = "standard") -> Dict[str, Any]:
        """Conduct comprehensive research on a query"""
        try:
            logger.info(f"Starting research: {query}")
            
            if not self.orchestrator:
                await self.initialize_components()
            
            # Configure research parameters
            depth_config = {
                "quick": {"max_iterations": 2, "min_findings": 5},
                "standard": {"max_iterations": 3, "min_findings": 10},
                "deep": {"max_iterations": 4, "min_findings": 15},
                "comprehensive": {"max_iterations": 5, "min_findings": 20}
            }
            
            config = depth_config.get(research_depth.lower(), depth_config["standard"])
            
            # Conduct research
            result = await self.orchestrator.orchestrate_research(
                query,
                max_iterations=config["max_iterations"],
                min_findings_threshold=config["min_findings"]
            )
            
            if result['status'] != 'success':
                logger.error(f"Research failed: {result['error']}")
                return result
            
            research_result = result['result']
            logger.info(f"Research completed - {len(research_result.findings)} findings")
            
            return {
                "status": "success",
                "research_result": research_result,
                "summary": {
                    "query": query,
                    "findings_count": len(research_result.findings),
                    "confidence_score": research_result.confidence_score,
                    "sources_count": len(research_result.sources_used)
                }
            }
            
        except Exception as e:
            logger.error(f"Error conducting research: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_web_interface(self):
        """Launch the Streamlit web interface"""
        try:
            logger.info("Launching web interface...")
            interface = InteractiveResearchInterface()
            interface.run()
        except Exception as e:
            logger.error(f"Error launching web interface: {e}")
            raise
    
    async def run_cli_interface(self):
        """Launch the CLI interface"""
        try:
            logger.info("Launching CLI interface...")
            await self.initialize_components()
            
            cli = CLIInterface()
            await cli.interactive_session()
        except Exception as e:
            logger.error(f"Error launching CLI interface: {e}")
            raise


async def main():
    """Main application entry point for CLI"""
    # Initialize agent for CLI
    agent = DeepResearchAgent()
    
    try:
        # Launch CLI interface
        await agent.run_cli_interface()
            
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check command line arguments first
    import argparse
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("--interface", choices=["cli", "web"], default="web",
                       help="Interface type (cli or web)")
    args = parser.parse_args()
    
    if args.interface == "web":
        # Running in Streamlit context
        agent = DeepResearchAgent()
        agent.run_web_interface()
    else:
        # Running as CLI application
        asyncio.run(main())