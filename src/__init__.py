"""
Deep Research Agent - A comprehensive local research system using LangChain, ChromaDB, and Ollama.
"""

__version__ = "1.0.0"
__author__ = "Deep Research Agent Team"
__email__ = "contact@deepresearch.ai"

from .embeddings.ollama_embeddings import OllamaEmbeddingService
from .retrieval.vector_store import VectorStoreManager
from .agents.research_agent import ResearchAgent
from .agents.manager_agent import ManagerAgent
from .chat.chat_interface import ChatInterface

__all__ = [
    "OllamaEmbeddingService",
    "VectorStoreManager", 
    "ResearchAgent",
    "ManagerAgent",
    "ChatInterface"
]