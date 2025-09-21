"""
Advanced Embedding and Vector Store System for Deep Researcher Agent
Provides local embedding generation and efficient retrieval with ChromaDB
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

# ChromaDB imports
import chromadb
from chromadb.config import Settings

from config import ResearchConfig
from document_processor import AdvancedDocumentProcessor, DocumentMetadata


@dataclass
class EmbeddingMetadata:
    """Metadata for embedding operations"""
    model_name: str
    dimension: int
    created_at: str
    document_count: int
    total_chunks: int
    
    
class AdvancedEmbeddingSystem:
    """Advanced embedding system with local Ollama models and ChromaDB"""
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.config.create_directories()
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.config.EMBEDDING_MODEL
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Document processor
        self.document_processor = AdvancedDocumentProcessor(config)
        
        # Metadata storage
        self.metadata_file = Path(self.config.VECTOR_DB_DIR) / "embedding_metadata.json"
        self.document_metadata_file = Path(self.config.VECTOR_DB_DIR) / "document_metadata.json"
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.embedding_metadata = json.load(f)
            else:
                self.embedding_metadata = {}
            
            if self.document_metadata_file.exists():
                with open(self.document_metadata_file, 'r') as f:
                    self.document_metadata = json.load(f)
            else:
                self.document_metadata = {}
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            self.embedding_metadata = {}
            self.document_metadata = {}
    
    def _save_metadata(self):
        """Save metadata to files"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.embedding_metadata, f, indent=2)
            
            with open(self.document_metadata_file, 'w') as f:
                json.dump(self.document_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def _check_document_changes(self, file_path: str, doc_metadata: DocumentMetadata) -> bool:
        """Check if document has changed since last processing"""
        if file_path not in self.document_metadata:
            return True
        
        stored_metadata = self.document_metadata[file_path]
        return stored_metadata.get('doc_hash') != doc_metadata.doc_hash
    
    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for documents"""
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_documents, texts
            )
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        try:
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            return embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise
    
    async def add_documents_from_file(self, file_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Add documents from a file to the vector store"""
        try:
            # Process the document
            documents, doc_metadata = await self.document_processor.process_file(file_path)
            
            # Check if document needs reprocessing
            if not force_reprocess and not self._check_document_changes(file_path, doc_metadata):
                print(f"Document {file_path} unchanged, skipping...")
                return {"status": "skipped", "reason": "unchanged"}
            
            # Add to vector store
            document_ids = [doc.metadata.get('chunk_id', f"doc_{i}") for i, doc in enumerate(documents)]
            
            self.vector_store.add_documents(
                documents=documents,
                ids=document_ids
            )
            
            # Update metadata
            self.document_metadata[file_path] = asdict(doc_metadata)
            self._save_metadata()
            
            print(f"Added {len(documents)} document chunks from {file_path}")
            
            return {
                "status": "success",
                "document_count": len(documents),
                "metadata": asdict(doc_metadata)
            }
            
        except Exception as e:
            print(f"Error adding documents from {file_path}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def add_documents_from_directory(self, directory_path: str, recursive: bool = True, force_reprocess: bool = False) -> Dict[str, Any]:
        """Add all documents from a directory to the vector store"""
        try:
            # Process all documents in directory
            documents, metadata_list = await self.document_processor.process_directory(
                directory_path, recursive
            )
            
            if not documents:
                return {"status": "no_documents", "message": "No documents found or processed"}
            
            # Filter documents that need processing
            if not force_reprocess:
                filtered_docs = []
                filtered_metadata = []
                
                for doc_meta in metadata_list:
                    if self._check_document_changes(doc_meta.file_path, doc_meta):
                        # Find all chunks for this document
                        doc_chunks = [doc for doc in documents 
                                    if doc.metadata.get('file_path') == doc_meta.file_path]
                        filtered_docs.extend(doc_chunks)
                        filtered_metadata.append(doc_meta)
                
                documents = filtered_docs
                metadata_list = filtered_metadata
            
            if not documents:
                print("All documents are up to date")
                return {"status": "up_to_date", "message": "All documents are current"}
            
            # Add to vector store in batches
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = [doc.metadata.get('chunk_id', f"doc_{i + j}") for j, doc in enumerate(batch)]
                
                self.vector_store.add_documents(
                    documents=batch,
                    ids=batch_ids
                )
                
                total_added += len(batch)
                print(f"Processed batch {i // batch_size + 1}: {total_added}/{len(documents)} documents")
            
            # Update metadata
            for doc_meta in metadata_list:
                self.document_metadata[doc_meta.file_path] = asdict(doc_meta)
            
            self._save_metadata()
            
            return {
                "status": "success",
                "total_documents": total_added,
                "files_processed": len(metadata_list),
                "metadata": [asdict(meta) for meta in metadata_list]
            }
            
        except Exception as e:
            print(f"Error adding documents from directory {directory_path}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> Dict[str, Any]:
        """Add documents directly from text strings"""
        try:
            from langchain_core.documents import Document
            
            # Create Document objects
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                metadata.update({
                    'chunk_id': f'text_doc_{i}_{hash(text) % 1000000}',
                    'source': metadata.get('source', f'direct_input_{i}'),
                    'chunk_index': i
                })
                
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
            
            # Add to vector store
            document_ids = [doc.metadata['chunk_id'] for doc in documents]
            
            self.vector_store.add_documents(
                documents=documents,
                ids=document_ids
            )
            
            print(f"Added {len(documents)} text documents to vector store")
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "document_ids": document_ids
            }
            
        except Exception as e:
            print(f"Error adding text documents: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_retriever(self, 
                     search_type: str = "similarity",
                     search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
        """Get a retriever for the vector store"""
        if search_kwargs is None:
            search_kwargs = {"k": self.config.DEFAULT_SEARCH_K}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    async def similarity_search(self, 
                              query: str, 
                              k: int = None,
                              filter_metadata: Optional[Dict[str, Any]] = None,
                              score_threshold: Optional[float] = None) -> List[Document]:
        """Perform similarity search"""
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            if score_threshold:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
                # Filter by score threshold
                filtered_results = [doc for doc, score in results if score >= score_threshold]
                return filtered_results
            else:
                return self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def similarity_search_with_score(self,
                                         query: str,
                                         k: int = None,
                                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Perform similarity search and return documents with scores"""
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            return results
        except Exception as e:
            print(f"Error in similarity search with score: {e}")
            return []
    
    async def max_marginal_relevance_search(self,
                                          query: str,
                                          k: int = None,
                                          fetch_k: int = None,
                                          lambda_mult: float = 0.5,
                                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform maximum marginal relevance search to reduce redundancy"""
        k = k or self.config.DEFAULT_SEARCH_K
        fetch_k = fetch_k or min(k * 3, self.config.MAX_DOCUMENTS_PER_QUERY)
        
        try:
            return self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_metadata
            )
        except Exception as e:
            print(f"Error in MMR search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            collection = self.chroma_client.get_collection(self.config.CHROMA_COLLECTION_NAME)
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.config.CHROMA_COLLECTION_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "files_indexed": len(self.document_metadata),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Could not get collection stats: {e}"}
    
    def delete_documents_by_source(self, source_path: str) -> bool:
        """Delete all documents from a specific source"""
        try:
            # Get all documents with this source
            results = self.vector_store.get(
                where={"file_path": source_path}
            )
            
            if results and results.get('ids'):
                self.vector_store.delete(ids=results['ids'])
                
                # Remove from metadata
                if source_path in self.document_metadata:
                    del self.document_metadata[source_path]
                    self._save_metadata()
                
                print(f"Deleted {len(results['ids'])} documents from {source_path}")
                return True
            else:
                print(f"No documents found for source: {source_path}")
                return False
                
        except Exception as e:
            print(f"Error deleting documents from {source_path}: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.config.CHROMA_COLLECTION_NAME)
            
            # Recreate the collection
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.config.CHROMA_PERSIST_DIRECTORY
            )
            
            # Clear metadata
            self.embedding_metadata = {}
            self.document_metadata = {}
            self._save_metadata()
            
            print("Collection reset successfully")
            return True
            
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False


# Enhanced retriever with multiple strategies
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining multiple search strategies"""
    
    embedding_system: AdvancedEmbeddingSystem
    config: ResearchConfig
    
    def __init__(self, embedding_system: AdvancedEmbeddingSystem, config: ResearchConfig = None):
        super().__init__(
            embedding_system=embedding_system,
            config=config or ResearchConfig()
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid approach"""
        return asyncio.run(self._aget_relevant_documents(query))
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of getting relevant documents"""
        try:
            # Strategy 1: Similarity search
            similarity_docs = await self.embedding_system.similarity_search(
                query=query,
                k=self.config.DEFAULT_SEARCH_K // 2
            )
            
            # Strategy 2: MMR search for diversity
            mmr_docs = await self.embedding_system.max_marginal_relevance_search(
                query=query,
                k=self.config.DEFAULT_SEARCH_K // 2,
                lambda_mult=0.7
            )
            
            # Combine results and remove duplicates
            all_docs = similarity_docs + mmr_docs
            unique_docs = []
            seen_ids = set()
            
            for doc in all_docs:
                doc_id = doc.metadata.get('chunk_id', doc.page_content[:100])
                if doc_id not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc_id)
            
            # Limit final results
            return unique_docs[:self.config.DEFAULT_SEARCH_K]
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return []


# Usage example and testing
if __name__ == "__main__":
    async def main():
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Test adding documents from existing CSV
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Result: {result}")
        
        # Test search
        if result.get("status") == "success":
            docs = await embedding_system.similarity_search("good pizza restaurant", k=3)
            print(f"\nFound {len(docs)} documents")
            for i, doc in enumerate(docs):
                print(f"\nDocument {i + 1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        
        # Get collection stats
        stats = embedding_system.get_collection_stats()
        print(f"\nCollection stats: {stats}")
    
    # Run the async function
    asyncio.run(main())