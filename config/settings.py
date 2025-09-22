"""
Simplified configuration for Deep Research Agent
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Ensure directories exist
for dir_path in [DATA_DIR, UPLOADS_DIR, EXPORTS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ollama Configuration - Simplified for local processing
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "chat_model": "llama3.2",
    "reasoning_model": "llama3.2:8b",
    "temperature": 0.1,
    "max_tokens": 4096,
    "timeout": 60
}

# ChromaDB Configuration - Simplified
CHROMA_CONFIG = {
    "persist_directory": str(VECTOR_DB_DIR),
    "collection_name": "research_documents",
    "distance_metric": "cosine",
    "n_results": 10,
    "similarity_threshold": 0.7
}

# Document Processing - Multi-format support
DOCUMENT_CONFIG = {
    "supported_formats": [".pdf", ".txt", ".docx", ".md", ".csv"],
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_file_size_mb": 50,
    "encoding": "utf-8"
}

# Agent Configuration - Simplified multi-agent setup
AGENT_CONFIG = {
    "manager": {
        "model": OLLAMA_CONFIG["chat_model"],
        "temperature": 0.1,
        "max_iterations": 3
    },
    "researcher": {
        "model": OLLAMA_CONFIG["reasoning_model"],
        "temperature": 0.2,
        "max_retrieval_docs": 10
    },
    "reasoning_depth": 3
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Deep Research Agent",
    "page_icon": "",
    "layout": "wide",
    "max_file_uploads": 10
}

# Export Configuration
EXPORT_CONFIG = {
    "include_citations": True,
    "include_reasoning_steps": True,
    "max_export_size_mb": 100
}

# Logging Configuration - Simplified
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time} | {level} | {name}:{function}:{line} - {message}",
    "rotation": "10 MB",
    "retention": "3 days"
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "ollama": OLLAMA_CONFIG,
        "chroma": CHROMA_CONFIG,
        "document": DOCUMENT_CONFIG,
        "agent": AGENT_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
        "export": EXPORT_CONFIG,
        "logging": LOGGING_CONFIG,
        "directories": {
            "base": BASE_DIR,
            "data": DATA_DIR,
            "uploads": UPLOADS_DIR,
            "exports": EXPORTS_DIR,
            "vector_db": VECTOR_DB_DIR,
        }
    }


def validate_ollama_connection() -> bool:
    """Simple validation for Ollama connection."""
    try:
        import httpx
        response = httpx.get(f"{OLLAMA_CONFIG['base_url']}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False