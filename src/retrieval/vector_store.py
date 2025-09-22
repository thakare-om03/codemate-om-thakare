"""
Vector Store Manager for ChromaDB integration
"""
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..embeddings.ollama_embeddings import OllamaEmbeddingService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChromaEmbeddingFunction(EmbeddingFunction):
    """
    Wrapper to make OllamaEmbeddingService compatible with ChromaDB.
    """
    
    def __init__(self, embedding_service: OllamaEmbeddingService):
        self.embedding_service = embedding_service
    
    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        return self.embedding_service.embed_documents(input_texts)


class VectorStoreManager:
    """
    Manager for ChromaDB vector store operations with dynamic collection management.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "research_documents",
        embedding_service: Optional[OllamaEmbeddingService] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the default collection
            embedding_service: Ollama embedding service instance
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = ChromaEmbeddingFunction(self.embedding_service)
        
        # Collection cache
        self._collections: Dict[str, Any] = {}
        
        logger.info(f"Initialized VectorStoreManager with persist directory: {self.persist_directory}")
    
    def create_collection(self, name: Optional[str] = None) -> str:
        """
        Create a new collection.
        
        Args:
            name: Collection name (generates UUID if None)
            
        Returns:
            Collection name
        """
        collection_name = name or f"collection_{uuid.uuid4().hex[:8]}"
        
        try:
            # Check if collection already exists
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name in existing_collections:
                logger.warning(f"Collection '{collection_name}' already exists")
                return collection_name
            
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            self._collections[collection_name] = collection
            logger.info(f"Created collection: {collection_name}")
            return collection_name
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    def get_collection(self, name: Optional[str] = None):
        """
        Get an existing collection.
        
        Args:
            name: Collection name (uses default if None)
            
        Returns:
            ChromaDB collection object
        """
        collection_name = name or self.collection_name
        
        try:
            if collection_name in self._collections:
                return self._collections[collection_name]
            
            # Get existing collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            self._collections[collection_name] = collection
            logger.debug(f"Retrieved collection: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}': {e}")
            # Try to create if it doesn't exist
            logger.info(f"Attempting to create collection '{collection_name}'")
            self.create_collection(collection_name)
            return self.get_collection(collection_name)
    
    def add_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            collection_name: Target collection name
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        try:
            collection = self.get_collection(collection_name)
            document_ids = []
            
            logger.info(f"Adding {len(documents)} documents to collection '{collection.name}'")
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                texts = [doc.page_content for doc in batch]
                metadatas = []
                ids = []
                
                for doc in batch:
                    doc_id = doc.metadata.get('id', str(uuid.uuid4()))
                    ids.append(doc_id)
                    
                    # Prepare metadata
                    metadata = dict(doc.metadata)
                    metadata.update({
                        'added_at': datetime.now().isoformat(),
                        'text_length': len(doc.page_content)
                    })
                    metadatas.append(metadata)
                
                # Add to collection
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                document_ids.extend(ids)
                logger.debug(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            logger.info(f"Successfully added {len(document_ids)} documents")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        include_scores: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            collection_name: Collection to search in
            n_results: Number of results to return
            where: Metadata filter conditions
            include_scores: Whether to include similarity scores
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            collection = self.get_collection(collection_name)
            
            logger.debug(f"Searching for: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to Document objects
            documents_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] or [{}] * len(results['documents'][0]),
                    results['distances'][0] or [0.0] * len(results['documents'][0])
                )):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    score = 1.0 - distance if distance is not None else 0.0
                    
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata or {}
                    )
                    
                    documents_with_scores.append((document, score))
            
            logger.debug(f"Found {len(documents_with_scores)} similar documents")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=name)
            
            # Remove from cache
            if name in self._collections:
                del self._collections[name]
            
            logger.info(f"Deleted collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            names = [col.name for col in collections]
            logger.debug(f"Found {len(names)} collections: {names}")
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.get_collection(name)
            count = collection.count()
            
            info = {
                "name": collection.name,
                "document_count": count,
                "metadata": collection.metadata or {}
            }
            
            logger.debug(f"Collection info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"name": name or self.collection_name, "document_count": 0, "metadata": {}, "error": str(e)}
    
    def clear_collection(self, name: Optional[str] = None) -> bool:
        """
        Clear all documents from a collection without deleting the collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_name = name or self.collection_name
            
            # Delete and recreate collection
            self.delete_collection(collection_name)
            self.create_collection(collection_name)
            
            logger.info(f"Cleared collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection '{collection_name}': {e}")
            return False
    
    def reset_database(self) -> bool:
        """
        Reset the entire vector database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Resetting entire vector database")
            
            # Clear collections cache
            self._collections.clear()
            
            # Reset ChromaDB
            self.client.reset()
            
            # Recreate default collection
            self.create_collection(self.collection_name)
            
            logger.info("Successfully reset vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False
    
    def delete_database(self) -> bool:
        """
        Completely delete the vector database from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Deleting vector database from disk")
            
            # Clear collections cache
            self._collections.clear()
            
            # Delete the persist directory
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            
            # Reinitialize
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Recreate default collection
            self.create_collection(self.collection_name)
            
            logger.info("Successfully deleted and reinitialized vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete database: {e}")
            return False
    
    def get_langchain_vectorstore(self, collection_name: Optional[str] = None) -> Chroma:
        """
        Get a LangChain Chroma vectorstore instance.
        
        Args:
            collection_name: Collection name
            
        Returns:
            LangChain Chroma vectorstore
        """
        try:
            collection_name = collection_name or self.collection_name
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_service,
                persist_directory=str(self.persist_directory)
            )
            
            logger.debug(f"Created LangChain vectorstore for collection: {collection_name}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create LangChain vectorstore: {e}")
            raise