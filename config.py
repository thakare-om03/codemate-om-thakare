"""
Configuration file for Deep Researcher Agent
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

class ResearchConfig:
    """Configuration class for the Deep Researcher Agent"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_DB_DIR = BASE_DIR / "vector_db"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Ollama model configurations
    DEFAULT_LLM_MODEL = "llama3.2"
    EMBEDDING_MODEL = "mxbai-embed-large"
    REASONING_MODEL = "llama3.2"  # For complex reasoning tasks
    SUMMARIZATION_MODEL = "llama3.2"  # For summarization tasks
    
    # ChromaDB configuration
    CHROMA_COLLECTION_NAME = "research_documents"
    CHROMA_PERSIST_DIRECTORY = str(VECTOR_DB_DIR)
    
    # Retrieval configuration
    DEFAULT_SEARCH_K = 10
    MAX_DOCUMENTS_PER_QUERY = 20
    SIMILARITY_THRESHOLD = 0.7
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_DOCUMENT_SIZE = 50_000_000  # 50MB
    
    # Research agent configuration
    MAX_RESEARCH_ITERATIONS = 5
    MAX_SUBQUERIES = 8
    MIN_SOURCES_PER_TOPIC = 3
    
    # Report generation
    DEFAULT_REPORT_FORMAT = "markdown"
    SUPPORTED_FORMATS = ["markdown", "pdf", "html"]
    MAX_REPORT_LENGTH = 10000
    
    # Quality control
    MIN_CONFIDENCE_SCORE = 0.6
    ENABLE_SOURCE_VALIDATION = True
    REQUIRE_MULTIPLE_SOURCES = True
    
    # Ollama model configurations as dict (for compatibility)
    OLLAMA_MODELS = {
        "llm": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 2048,
        },
        "reasoning": {
            "model": REASONING_MODEL,
            "temperature": 0.0,
            "top_p": 0.8,
            "num_predict": 4096,
        },
        "summarization": {
            "model": SUMMARIZATION_MODEL,
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 1024,
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
        }
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.DOCUMENTS_DIR, cls.VECTOR_DB_DIR, cls.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_ollama_config(cls) -> Dict[str, Any]:
        """Get Ollama configuration for different models"""
        return cls.OLLAMA_MODELS