"""
Ollama Embedding Service for local embedding generation
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from ..utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbeddingService:
    """
    Service for generating embeddings using Ollama models locally.
    Provides high-performance embedding generation for document indexing and retrieval.
    """
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize the Ollama embedding service.
        
        Args:
            model: The Ollama embedding model to use
            base_url: The base URL for the Ollama server
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.base_url = base_url
        self.config = kwargs
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=model,
                base_url=base_url,
                **kwargs
            )
            logger.info(f"Initialized Ollama embeddings with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            logger.debug(f"Generating query embedding for: {text[:100]}...")
            embedding = self.embeddings.embed_query(text)
            logger.debug("Successfully generated query embedding")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generate embeddings for documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings asynchronously for {len(texts)} documents")
            # Run embedding generation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.embeddings.embed_documents, texts
            )
            logger.info(f"Successfully generated {len(embeddings)} embeddings asynchronously")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate document embeddings asynchronously: {e}")
            raise
    
    async def embed_query_async(self, text: str) -> List[float]:
        """
        Asynchronously generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            logger.debug(f"Generating query embedding asynchronously for: {text[:100]}...")
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embeddings.embed_query, text
            )
            logger.debug("Successfully generated query embedding asynchronously")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding asynchronously: {e}")
            raise
    
    def embed_documents_batch(
        self,
        texts: List[str],
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Generate embeddings for documents in batches for better performance.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings in batches of {batch_size} for {len(texts)} documents")
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings in batches")
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to generate batched embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        try:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_query("test")
            dimension = len(test_embedding)
            logger.info(f"Embedding dimension: {dimension}")
            return dimension
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")
            # Default dimension for nomic-embed-text
            return 768
    
    def validate_model_availability(self) -> bool:
        """
        Check if the Ollama model is available and responding.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            logger.info(f"Validating model availability: {self.model}")
            # Try to generate a test embedding
            test_embedding = self.embed_query("test connection")
            is_available = len(test_embedding) > 0
            
            if is_available:
                logger.info(f"Model {self.model} is available and responding")
            else:
                logger.warning(f"Model {self.model} responded but returned empty embedding")
            
            return is_available
        except Exception as e:
            logger.error(f"Model {self.model} is not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            dimension = self.get_embedding_dimension()
            is_available = self.validate_model_availability()
            
            info = {
                "model_name": self.model,
                "base_url": self.base_url,
                "embedding_dimension": dimension,
                "is_available": is_available,
                "config": self.config
            }
            
            logger.info(f"Model info: {info}")
            return info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "model_name": self.model,
                "base_url": self.base_url,
                "embedding_dimension": 768,  # Default
                "is_available": False,
                "config": self.config,
                "error": str(e)
            }


class EmbeddingManager:
    """
    Manager class for handling multiple embedding services and models.
    """
    
    def __init__(self):
        self.services: Dict[str, OllamaEmbeddingService] = {}
        self.default_service: Optional[str] = None
        
    def add_service(
        self,
        name: str,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        make_default: bool = False,
        **kwargs
    ) -> OllamaEmbeddingService:
        """
        Add a new embedding service.
        
        Args:
            name: Name for the service
            model: Ollama model to use
            base_url: Ollama server URL
            make_default: Whether to make this the default service
            **kwargs: Additional configuration
            
        Returns:
            The created embedding service
        """
        try:
            service = OllamaEmbeddingService(
                model=model,
                base_url=base_url,
                **kwargs
            )
            
            self.services[name] = service
            
            if make_default or self.default_service is None:
                self.default_service = name
                
            logger.info(f"Added embedding service '{name}' with model '{model}'")
            return service
        except Exception as e:
            logger.error(f"Failed to add embedding service '{name}': {e}")
            raise
    
    def get_service(self, name: Optional[str] = None) -> OllamaEmbeddingService:
        """
        Get an embedding service by name.
        
        Args:
            name: Service name (uses default if None)
            
        Returns:
            The requested embedding service
        """
        service_name = name or self.default_service
        
        if service_name is None:
            raise ValueError("No embedding service specified and no default service set")
        
        if service_name not in self.services:
            raise ValueError(f"Embedding service '{service_name}' not found")
        
        return self.services[service_name]
    
    def list_services(self) -> List[str]:
        """Get list of available service names."""
        return list(self.services.keys())
    
    def validate_all_services(self) -> Dict[str, bool]:
        """
        Validate all registered embedding services.
        
        Returns:
            Dictionary mapping service names to availability status
        """
        results = {}
        for name, service in self.services.items():
            try:
                results[name] = service.validate_model_availability()
            except Exception as e:
                logger.error(f"Failed to validate service '{name}': {e}")
                results[name] = False
        
        return results