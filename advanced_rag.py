"""
Advanced RAG (Retrieval-Augmented Generation) patterns for the Deep Research Agent.
Implements latest retrieval strategies including contextual compression, 
multi-vector retrieval, and parent document retrieval.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import (
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
    MultiVectorRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_ollama import ChatOllama

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem


class AdvancedRetriever:
    """
    Advanced retriever implementing multiple modern RAG strategies.
    Based on latest LangChain patterns for improved retrieval quality.
    """
    
    def __init__(self, embedding_system: AdvancedEmbeddingSystem, config: ResearchConfig = None):
        self.embedding_system = embedding_system
        self.config = config or ResearchConfig()
        
        # Initialize LLM for compression
        llm_config = self.config.get_ollama_config()
        self.llm = ChatOllama(**llm_config["llm"])
        
        # Initialize retrievers
        self._init_retrievers()
    
    def _init_retrievers(self):
        """Initialize different retriever types"""
        # Base retriever
        self.base_retriever = self.embedding_system.get_retriever()
        
        # Contextual compression retriever
        self.compression_retriever = self._create_compression_retriever()
        
        # Multi-vector retriever for different granularities
        self.multi_vector_retriever = self._create_multi_vector_retriever()
        
        # Parent document retriever for context preservation
        self.parent_doc_retriever = self._create_parent_document_retriever()
    
    def _create_compression_retriever(self) -> ContextualCompressionRetriever:
        """Create contextual compression retriever"""
        # LLM chain extractor for compression
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def _create_multi_vector_retriever(self) -> MultiVectorRetriever:
        """Create multi-vector retriever for different text granularities"""
        # In-memory store for document mappings
        store = InMemoryStore()
        
        # Text splitters for different granularities
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        return MultiVectorRetriever(
            vectorstore=self.embedding_system.vector_store,
            docstore=store,
            id_key="doc_id",
            search_kwargs={"k": self.config.DEFAULT_SEARCH_K}
        )
    
    def _create_parent_document_retriever(self) -> ParentDocumentRetriever:
        """Create parent document retriever for maintaining context"""
        # In-memory store for parent documents
        store = InMemoryStore()
        
        # Child splitter for small chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        return ParentDocumentRetriever(
            vectorstore=self.embedding_system.vector_store,
            docstore=store,
            child_splitter=child_splitter,
            search_kwargs={"k": self.config.DEFAULT_SEARCH_K}
        )
    
    async def hybrid_search(self, query: str, strategy: str = "adaptive", k: int = None) -> List[Document]:
        """
        Perform hybrid search using multiple retrieval strategies.
        
        Args:
            query: Search query
            strategy: Retrieval strategy ('adaptive', 'compression', 'multi_vector', 'parent_doc', 'fusion')
            k: Number of results to return
        
        Returns:
            List of retrieved documents
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        if strategy == "compression":
            return await self._compression_search(query, k)
        elif strategy == "multi_vector":
            return await self._multi_vector_search(query, k)
        elif strategy == "parent_doc":
            return await self._parent_document_search(query, k)
        elif strategy == "fusion":
            return await self._fusion_search(query, k)
        else:  # adaptive
            return await self._adaptive_search(query, k)
    
    async def _compression_search(self, query: str, k: int) -> List[Document]:
        """Contextual compression search"""
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.compression_retriever.get_relevant_documents, query
            )
            return docs[:k]
        except Exception as e:
            print(f"Error in compression search: {e}")
            # Fallback to base retriever
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def _multi_vector_search(self, query: str, k: int) -> List[Document]:
        """Multi-vector search across different granularities"""
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.multi_vector_retriever.get_relevant_documents, query
            )
            return docs[:k]
        except Exception as e:
            print(f"Error in multi-vector search: {e}")
            # Fallback to base retriever
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def _parent_document_search(self, query: str, k: int) -> List[Document]:
        """Parent document search for full context"""
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.parent_doc_retriever.get_relevant_documents, query
            )
            return docs[:k]
        except Exception as e:
            print(f"Error in parent document search: {e}")
            # Fallback to base retriever
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def _fusion_search(self, query: str, k: int) -> List[Document]:
        """
        Fusion search combining multiple retrieval strategies.
        Uses reciprocal rank fusion to combine results.
        """
        try:
            # Get results from different strategies
            tasks = [
                self.embedding_system.similarity_search(query, k=k),
                self._compression_search(query, k),
                self.embedding_system.max_marginal_relevance_search(query, k=k)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            if not valid_results:
                return await self.embedding_system.similarity_search(query, k=k)
            
            # Apply reciprocal rank fusion
            return self._reciprocal_rank_fusion(valid_results, k)
            
        except Exception as e:
            print(f"Error in fusion search: {e}")
            return await self.embedding_system.similarity_search(query, k=k)
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[Document]], k: int) -> List[Document]:
        """
        Combine multiple result lists using reciprocal rank fusion.
        
        Args:
            result_lists: List of document lists from different retrievers
            k: Number of final results to return
        
        Returns:
            Fused and ranked document list
        """
        # Create a mapping of documents to their fusion scores
        doc_scores = {}
        doc_map = {}  # To store unique documents
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                # Use content hash as unique identifier
                doc_id = hash(doc.page_content)
                doc_map[doc_id] = doc
                
                # Calculate reciprocal rank fusion score
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                
                doc_scores[doc_id] += 1 / (rank + 60)  # 60 is a common constant for RRF
        
        # Sort documents by fusion score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc_map[doc_id] for doc_id, score in sorted_docs[:k]]
    
    async def _adaptive_search(self, query: str, k: int) -> List[Document]:
        """
        Adaptive search that chooses the best strategy based on query characteristics.
        """
        try:
            # Analyze query characteristics
            query_length = len(query.split())
            is_complex = query_length > 10 or any(word in query.lower() 
                                                for word in ['compare', 'analyze', 'explain', 'how', 'why'])
            
            # Choose strategy based on query complexity
            if is_complex:
                # Use fusion search for complex queries
                return await self._fusion_search(query, k)
            else:
                # Use compression search for simple queries
                return await self._compression_search(query, k)
                
        except Exception as e:
            print(f"Error in adaptive search: {e}")
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def semantic_similarity_search(self, query: str, threshold: float = 0.7, k: int = None) -> List[Tuple[Document, float]]:
        """
        Semantic similarity search with score filtering.
        
        Args:
            query: Search query
            threshold: Minimum similarity threshold
            k: Maximum number of results
        
        Returns:
            List of (document, score) tuples
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Get documents with scores
            docs_with_scores = await self.embedding_system.similarity_search_with_score(query, k=k * 2)
            
            # Filter by threshold
            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= threshold]
            
            return filtered_docs[:k]
            
        except Exception as e:
            print(f"Error in semantic similarity search: {e}")
            # Fallback without scores
            docs = await self.embedding_system.similarity_search(query, k=k)
            return [(doc, 1.0) for doc in docs]
    
    async def contextual_retrieval(self, query: str, context: str = None, k: int = None) -> List[Document]:
        """
        Contextual retrieval that considers conversation context.
        
        Args:
            query: Current query
            context: Previous conversation context
            k: Number of results
        
        Returns:
            Context-aware retrieved documents
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Enhance query with context if provided
            if context:
                enhanced_query = f"Context: {context}\\n\\nQuery: {query}"
            else:
                enhanced_query = query
            
            # Use adaptive search for context-aware retrieval
            return await self._adaptive_search(enhanced_query, k)
            
        except Exception as e:
            print(f"Error in contextual retrieval: {e}")
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def multi_modal_retrieval(self, text_query: str = None, image_query: Any = None, k: int = None) -> List[Document]:
        """
        Multi-modal retrieval for text and image queries.
        Note: Currently text-only, placeholder for future image support.
        
        Args:
            text_query: Text-based query
            image_query: Image-based query (future enhancement)
            k: Number of results
        
        Returns:
            Retrieved documents
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        if text_query:
            return await self.hybrid_search(text_query, strategy="adaptive", k=k)
        else:
            # Future: implement image-based retrieval
            raise NotImplementedError("Image-based retrieval not yet implemented")
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        return {
            "total_documents": self.embedding_system.vector_store._collection.count(),
            "embedding_model": self.config.EMBEDDING_MODEL,
            "vector_dimensions": len(self.embedding_system.embeddings.embed_query("test")),
            "available_strategies": [
                "similarity", "compression", "multi_vector", 
                "parent_doc", "fusion", "adaptive"
            ],
            "default_k": self.config.DEFAULT_SEARCH_K
        }