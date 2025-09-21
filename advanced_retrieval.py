"""
Advanced Retrieval System with Multiple Strategies
Implements hybrid retrieval, multi-query generation, contextual compression, and result fusion
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np

# LangChain imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# Retrieval components
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem


class MultiQueryGeneration(BaseModel):
    """Model for generating multiple query variations"""
    queries: List[str] = Field(description="List of query variations for comprehensive search")
    focus_areas: List[str] = Field(description="Different focus areas or perspectives to explore")
    search_strategies: List[str] = Field(description="Recommended search strategies for each query")


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata"""
    document: Document
    score: float
    retrieval_method: str
    query_used: str
    rank: int
    relevance_explanation: str


@dataclass
class FusedResults:
    """Results from multiple retrieval strategies"""
    documents: List[Document]
    scores: List[float]
    methods_used: List[str]
    fusion_strategy: str
    total_unique_results: int


class QueryExpansionEngine:
    """Engine for expanding queries using various techniques"""
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=MultiQueryGeneration)
    
    async def generate_query_variations(self, original_query: str, num_variations: int = 5) -> MultiQueryGeneration:
        """Generate multiple variations of the original query"""
        try:
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert search query expansion specialist. Generate multiple variations of the given query to ensure comprehensive information retrieval.
                
                Create variations that:
                1. Use different wording and synonyms
                2. Focus on different aspects of the topic
                3. Include related concepts and broader contexts
                4. Consider different perspectives or viewpoints
                5. Use both specific and general terminology
                
                {format_instructions}"""),
                ("human", "Original Query: {query}\nNumber of variations needed: {num_variations}")
            ])
            
            formatted_prompt = expansion_prompt.format_messages(
                query=original_query,
                num_variations=num_variations,
                format_instructions=self.parser.get_format_instructions()
            )
            
            response = await self.llm.ainvoke(formatted_prompt)
            return self.parser.parse(response.content)
            
        except Exception as e:
            print(f"Error in query expansion: {e}")
            # Fallback to simple variations
            return MultiQueryGeneration(
                queries=[original_query],
                focus_areas=["main_topic"],
                search_strategies=["semantic_search"]
            )
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for keyword-based search"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'about', 'what', 'how', 'why', 'when', 'where',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords


class AdvancedRetriever:
    """Advanced retriever with multiple search strategies"""
    
    def __init__(self, embedding_system: AdvancedEmbeddingSystem, config: ResearchConfig):
        self.embedding_system = embedding_system
        self.config = config
        
        # Initialize LLM for query expansion and compression
        model_config = config.get_ollama_config()
        self.llm = ChatOllama(**model_config["llm"])
        
        # Initialize query expansion engine
        self.query_expander = QueryExpansionEngine(self.llm)
        
        # Initialize different retrieval strategies
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize different types of retrievers"""
        # Base semantic retriever
        self.semantic_retriever = self.embedding_system.get_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.DEFAULT_SEARCH_K}
        )
        
        # MMR retriever for diversity
        self.mmr_retriever = self.embedding_system.get_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.DEFAULT_SEARCH_K,
                "fetch_k": self.config.DEFAULT_SEARCH_K * 2,
                "lambda_mult": 0.7
            }
        )
        
        # Multi-query retriever
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.semantic_retriever,
            llm=self.llm
        )
        
        # Contextual compression retriever
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.semantic_retriever
        )
    
    async def _keyword_search(self, query: str, k: int = None) -> List[Document]:
        """Perform keyword-based search"""
        try:
            k = k or self.config.DEFAULT_SEARCH_K
            keywords = self.query_expander.extract_keywords(query)
            
            # Create keyword query
            keyword_query = " OR ".join(keywords) if keywords else query
            
            # Use semantic search as fallback for keyword search
            # In a real implementation, you might use a proper full-text search index
            docs = await self.embedding_system.similarity_search(
                query=keyword_query,
                k=k
            )
            
            # Score based on keyword matches
            scored_docs = []
            for doc in docs:
                content_lower = doc.page_content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
                score = keyword_matches / max(len(keywords), 1)
                
                # Add score to metadata
                doc.metadata["keyword_score"] = score
                scored_docs.append(doc)
            
            # Sort by keyword score
            scored_docs.sort(key=lambda x: x.metadata.get("keyword_score", 0), reverse=True)
            return scored_docs
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    async def _semantic_search(self, query: str, k: int = None) -> List[Document]:
        """Perform semantic similarity search"""
        try:
            k = k or self.config.DEFAULT_SEARCH_K
            return await self.embedding_system.similarity_search(query=query, k=k)
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    async def _mmr_search(self, query: str, k: int = None) -> List[Document]:
        """Perform Maximum Marginal Relevance search"""
        try:
            k = k or self.config.DEFAULT_SEARCH_K
            return await self.embedding_system.max_marginal_relevance_search(
                query=query,
                k=k,
                lambda_mult=0.7
            )
        except Exception as e:
            print(f"Error in MMR search: {e}")
            return []
    
    async def _multi_query_search(self, query: str, k: int = None) -> List[Document]:
        """Perform multi-query search"""
        try:
            k = k or self.config.DEFAULT_SEARCH_K
            
            # Generate query variations
            query_variations = await self.query_expander.generate_query_variations(query)
            
            all_docs = []
            for variant_query in query_variations.queries[:3]:  # Limit to top 3 variations
                docs = await self.embedding_system.similarity_search(
                    query=variant_query,
                    k=k // len(query_variations.queries[:3])
                )
                all_docs.extend(docs)
            
            # Remove duplicates based on content similarity
            unique_docs = self._remove_duplicate_documents(all_docs)
            return unique_docs[:k]
            
        except Exception as e:
            print(f"Error in multi-query search: {e}")
            return []
    
    async def _compressed_search(self, query: str, k: int = None) -> List[Document]:
        """Perform contextual compression search"""
        try:
            k = k or self.config.DEFAULT_SEARCH_K
            
            # Get more documents initially
            initial_docs = await self.embedding_system.similarity_search(
                query=query,
                k=k * 2
            )
            
            # Apply compression (extract relevant parts)
            compressed_docs = []
            for doc in initial_docs:
                # Simple compression: extract sentences that contain query keywords
                keywords = self.query_expander.extract_keywords(query)
                sentences = doc.page_content.split('.')
                
                relevant_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in keywords):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    compressed_content = '. '.join(relevant_sentences[:3])  # Top 3 sentences
                    compressed_doc = Document(
                        page_content=compressed_content,
                        metadata={**doc.metadata, "compression_applied": True}
                    )
                    compressed_docs.append(compressed_doc)
            
            return compressed_docs[:k]
            
        except Exception as e:
            print(f"Error in compressed search: {e}")
            return []
    
    def _remove_duplicate_documents(self, documents: List[Document], similarity_threshold: float = 0.8) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        if not documents:
            return documents
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Simple deduplication based on first 200 characters
            content_signature = doc.page_content[:200].strip().lower()
            
            if content_signature not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_signature)
        
        return unique_docs
    
    def _fusion_score_combination(self, results_list: List[List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        """Combine scores from multiple retrieval methods using fusion"""
        # Document ID to score mapping
        doc_scores = {}
        doc_objects = {}
        
        for method_idx, results in enumerate(results_list):
            method_weight = 1.0 / len(results_list)  # Equal weighting
            
            for rank, (doc, score) in enumerate(results):
                doc_id = doc.page_content[:100]  # Use content snippet as ID
                
                # Reciprocal rank fusion score
                rank_score = 1.0 / (rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_objects[doc_id] = doc
                
                doc_scores[doc_id] += method_weight * rank_score
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs]
    
    async def hybrid_search(self, 
                          query: str, 
                          k: int = None,
                          methods: List[str] = None,
                          fusion_strategy: str = "rrf") -> FusedResults:
        """Perform hybrid search using multiple retrieval strategies"""
        k = k or self.config.DEFAULT_SEARCH_K
        methods = methods or ["semantic", "keyword", "mmr", "multi_query"]
        
        # Available search methods
        search_methods = {
            "semantic": self._semantic_search,
            "keyword": self._keyword_search,
            "mmr": self._mmr_search,
            "multi_query": self._multi_query_search,
            "compressed": self._compressed_search
        }
        
        # Execute searches in parallel
        tasks = []
        valid_methods = []
        
        for method in methods:
            if method in search_methods:
                tasks.append(search_methods[method](query, k))
                valid_methods.append(method)
        
        # Wait for all searches to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for i, result in enumerate(results_list):
            if not isinstance(result, Exception) and result:
                # Convert to scored tuples
                scored_results = [(doc, 1.0 - (rank * 0.1)) for rank, doc in enumerate(result)]
                valid_results.append(scored_results)
        
        if not valid_results:
            return FusedResults(
                documents=[],
                scores=[],
                methods_used=[],
                fusion_strategy=fusion_strategy,
                total_unique_results=0
            )
        
        # Apply fusion strategy
        if fusion_strategy == "rrf":  # Reciprocal Rank Fusion
            fused_results = self._fusion_score_combination(valid_results)
        else:
            # Simple concatenation and deduplication
            all_docs = []
            for method_results in valid_results:
                all_docs.extend([doc for doc, score in method_results])
            
            unique_docs = self._remove_duplicate_documents(all_docs)
            fused_results = [(doc, 1.0) for doc in unique_docs]
        
        # Limit results
        final_results = fused_results[:k]
        
        return FusedResults(
            documents=[doc for doc, score in final_results],
            scores=[score for doc, score in final_results],
            methods_used=valid_methods,
            fusion_strategy=fusion_strategy,
            total_unique_results=len(final_results)
        )
    
    async def adaptive_search(self, query: str, context: str = None, k: int = None) -> FusedResults:
        """Adaptive search that chooses the best strategy based on query characteristics"""
        k = k or self.config.DEFAULT_SEARCH_K
        
        # Analyze query characteristics
        query_keywords = self.query_expander.extract_keywords(query)
        query_length = len(query.split())
        
        # Choose search strategy based on query characteristics
        if query_length <= 3 and len(query_keywords) <= 2:
            # Short query: use keyword and semantic search
            methods = ["keyword", "semantic"]
        elif query_length > 10:
            # Long query: use compression and multi-query
            methods = ["compressed", "multi_query", "semantic"]
        else:
            # Medium query: use hybrid approach
            methods = ["semantic", "mmr", "multi_query"]
        
        return await self.hybrid_search(query, k, methods, "rrf")
    
    async def research_focused_search(self, 
                                    query: str, 
                                    research_context: Dict[str, Any] = None,
                                    k: int = None) -> FusedResults:
        """Research-focused search optimized for comprehensive information gathering"""
        k = k or self.config.DEFAULT_SEARCH_K
        
        # Generate comprehensive query variations
        query_variations = await self.query_expander.generate_query_variations(
            query, num_variations=6
        )
        
        # Use all available methods for comprehensive coverage
        methods = ["semantic", "keyword", "mmr", "multi_query", "compressed"]
        
        # Perform hybrid search
        results = await self.hybrid_search(query, k * 2, methods, "rrf")
        
        # If we have research context, filter and re-rank based on that
        if research_context and results.documents:
            filtered_docs = self._filter_by_research_context(
                results.documents, 
                research_context
            )
            
            return FusedResults(
                documents=filtered_docs[:k],
                scores=results.scores[:len(filtered_docs[:k])],
                methods_used=results.methods_used,
                fusion_strategy="research_focused",
                total_unique_results=len(filtered_docs[:k])
            )
        
        # Limit to requested number
        return FusedResults(
            documents=results.documents[:k],
            scores=results.scores[:k],
            methods_used=results.methods_used,
            fusion_strategy="research_focused",
            total_unique_results=len(results.documents[:k])
        )
    
    def _filter_by_research_context(self, 
                                   documents: List[Document], 
                                   research_context: Dict[str, Any]) -> List[Document]:
        """Filter documents based on research context"""
        # This is a placeholder for more sophisticated filtering
        # You could implement filtering based on:
        # - Document age/recency
        # - Source credibility
        # - Content type
        # - Relevance to specific research goals
        
        keywords = research_context.get("keywords", [])
        if not keywords:
            return documents
        
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            
            # Add context score to metadata
            doc.metadata["context_score"] = keyword_matches / max(len(keywords), 1)
            scored_docs.append(doc)
        
        # Sort by context score
        scored_docs.sort(key=lambda x: x.metadata.get("context_score", 0), reverse=True)
        return scored_docs
    
    # Enhanced RAG Methods - Latest Patterns from LangChain
    
    async def contextual_compression_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform contextual compression search using LLM-based compression.
        This method extracts only the most relevant parts of documents.
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.compression_retriever.get_relevant_documents, query
            )
            return docs[:k]
        except Exception as e:
            print(f"Error in contextual compression search: {e}")
            # Fallback to semantic search
            return await self.embedding_system.similarity_search(query, k=k)
    
    async def reciprocal_rank_fusion_search(self, query: str, k: int = None) -> FusedResults:
        """
        Advanced fusion search using reciprocal rank fusion algorithm.
        Combines results from multiple retrieval strategies with sophisticated ranking.
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Get results from different strategies
            tasks = [
                self.embedding_system.similarity_search(query, k=k),
                self.contextual_compression_search(query, k=k),
                self.embedding_system.max_marginal_relevance_search(query, k=k)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            if not valid_results:
                fallback_docs = await self.embedding_system.similarity_search(query, k=k)
                return FusedResults(
                    documents=fallback_docs,
                    scores=[1.0] * len(fallback_docs),
                    methods_used=["fallback_similarity"],
                    fusion_strategy="reciprocal_rank_fusion",
                    total_unique_results=len(fallback_docs)
                )
            
            # Apply reciprocal rank fusion
            fused_docs = self._reciprocal_rank_fusion(valid_results, k)
            
            return FusedResults(
                documents=fused_docs,
                scores=[1.0] * len(fused_docs),  # RRF provides implicit scoring
                methods_used=["similarity", "compression", "mmr"],
                fusion_strategy="reciprocal_rank_fusion",
                total_unique_results=len(fused_docs)
            )
            
        except Exception as e:
            print(f"Error in reciprocal rank fusion search: {e}")
            fallback_docs = await self.embedding_system.similarity_search(query, k=k)
            return FusedResults(
                documents=fallback_docs,
                scores=[1.0] * len(fallback_docs),
                methods_used=["fallback"],
                fusion_strategy="simple",
                total_unique_results=len(fallback_docs)
            )
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[Document]], k: int) -> List[Document]:
        """
        Apply reciprocal rank fusion algorithm to combine multiple result lists.
        RRF formula: score = sum(1 / (rank + 60)) for each list containing the document
        """
        doc_scores = {}
        doc_map = {}
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                # Use content hash as unique identifier
                doc_id = hash(doc.page_content)
                doc_map[doc_id] = doc
                
                # Calculate RRF score
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                
                doc_scores[doc_id] += 1 / (rank + 60)  # 60 is the standard RRF constant
        
        # Sort by fusion score and return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, score in sorted_docs[:k]]
    
    async def adaptive_retrieval_strategy(self, query: str, k: int = None) -> FusedResults:
        """
        Intelligent retrieval strategy selection based on query characteristics.
        Analyzes the query to determine the optimal retrieval approach.
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Analyze query characteristics
            query_analysis = self._analyze_query_complexity(query)
            
            # Choose strategy based on analysis
            if query_analysis["is_complex"]:
                # Complex queries benefit from comprehensive search
                return await self.reciprocal_rank_fusion_search(query, k)
            elif query_analysis["is_comparative"]:
                # Comparative queries need diverse perspectives
                return await self.hybrid_search(query, k)
            elif query_analysis["is_factual"]:
                # Factual queries work well with compression
                docs = await self.contextual_compression_search(query, k)
                return FusedResults(
                    documents=docs,
                    scores=[1.0] * len(docs),
                    methods_used=["contextual_compression"],
                    fusion_strategy="adaptive_factual",
                    total_unique_results=len(docs)
                )
            else:
                # Default to multi-query approach
                docs = await self._multi_query_search(query, k)
                return FusedResults(
                    documents=docs,
                    scores=[1.0] * len(docs),
                    methods_used=["multi_query"],
                    fusion_strategy="adaptive_default",
                    total_unique_results=len(docs)
                )
                
        except Exception as e:
            print(f"Error in adaptive retrieval: {e}")
            fallback_docs = await self.embedding_system.similarity_search(query, k=k)
            return FusedResults(
                documents=fallback_docs,
                scores=[1.0] * len(fallback_docs),
                methods_used=["fallback"],
                fusion_strategy="adaptive_fallback",
                total_unique_results=len(fallback_docs)
            )
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine optimal retrieval strategy"""
        query_lower = query.lower()
        
        # Keywords indicating complexity
        complex_indicators = [
            "compare", "contrast", "analyze", "evaluate", "assess", "examine",
            "how does", "why does", "what are the differences", "pros and cons"
        ]
        
        comparative_indicators = [
            "versus", "vs", "compared to", "difference between", "better than",
            "compare", "contrast", "which is"
        ]
        
        factual_indicators = [
            "what is", "define", "definition", "meaning", "who is", "when did",
            "where is", "list"
        ]
        
        return {
            "is_complex": any(indicator in query_lower for indicator in complex_indicators),
            "is_comparative": any(indicator in query_lower for indicator in comparative_indicators),
            "is_factual": any(indicator in query_lower for indicator in factual_indicators),
            "word_count": len(query.split()),
            "has_questions": "?" in query
        }
    
    async def semantic_similarity_threshold_search(self, query: str, 
                                                 threshold: float = 0.7, 
                                                 k: int = None) -> List[Tuple[Document, float]]:
        """
        Semantic search with similarity threshold filtering.
        Only returns documents above the specified similarity threshold.
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Get documents with scores
            docs_with_scores = await self.embedding_system.similarity_search_with_score(query, k=k * 2)
            
            # Filter by threshold
            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= threshold]
            
            return filtered_docs[:k]
            
        except Exception as e:
            print(f"Error in threshold search: {e}")
            # Fallback without scores
            docs = await self.embedding_system.similarity_search(query, k=k)
            return [(doc, 1.0) for doc in docs]
    
    async def contextual_conversation_search(self, current_query: str, 
                                           conversation_history: List[str] = None,
                                           k: int = None) -> List[Document]:
        """
        Context-aware search that considers previous conversation turns.
        Enhances current query with conversation context for better retrieval.
        """
        k = k or self.config.DEFAULT_SEARCH_K
        
        try:
            # Build enhanced query with context
            if conversation_history and len(conversation_history) > 0:
                # Combine recent context (last 3 turns) with current query
                recent_context = " ".join(conversation_history[-3:])
                enhanced_query = f"Previous context: {recent_context}\\n\\nCurrent question: {current_query}"
            else:
                enhanced_query = current_query
            
            # Use adaptive strategy for context-aware search
            results = await self.adaptive_retrieval_strategy(enhanced_query, k)
            return results.documents
            
        except Exception as e:
            print(f"Error in contextual search: {e}")
            return await self.embedding_system.similarity_search(current_query, k=k)
    
    def get_advanced_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the advanced retrieval system"""
        try:
            base_stats = {
                "total_documents": self.embedding_system.vector_store._collection.count(),
                "embedding_model": self.config.EMBEDDING_MODEL,
                "default_k": self.config.DEFAULT_SEARCH_K
            }
            
            # Add advanced strategy information
            advanced_stats = {
                "available_strategies": [
                    "semantic_similarity", "mmr_diversity", "multi_query", 
                    "contextual_compression", "reciprocal_rank_fusion",
                    "adaptive_strategy", "threshold_filtering", "contextual_conversation"
                ],
                "fusion_algorithms": ["weighted_sum", "reciprocal_rank_fusion", "borda_count"],
                "compression_enabled": True,
                "multi_query_enabled": True,
                "adaptive_strategy_enabled": True
            }
            
            return {**base_stats, **advanced_stats}
            
        except Exception as e:
            print(f"Error getting retrieval stats: {e}")
            return {"error": str(e)}


# Usage example and testing
if __name__ == "__main__":
    async def main():
        from embedding_system import AdvancedEmbeddingSystem
        
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add documents first
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Document loading result: {result}")
        
        if result.get("status") == "success":
            # Create advanced retriever
            retriever = AdvancedRetriever(embedding_system, config)
            
            # Test different search strategies
            query = "best pizza restaurants with good service"
            
            print(f"\nTesting query: {query}")
            
            # Test hybrid search
            hybrid_results = await retriever.hybrid_search(query, k=5)
            print(f"\nHybrid Search Results:")
            print(f"Methods used: {hybrid_results.methods_used}")
            print(f"Total results: {hybrid_results.total_unique_results}")
            
            for i, doc in enumerate(hybrid_results.documents[:3]):
                print(f"\nResult {i+1} (Score: {hybrid_results.scores[i]:.3f}):")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
            
            # Test adaptive search
            adaptive_results = await retriever.adaptive_search(query, k=5)
            print(f"\nAdaptive Search Results:")
            print(f"Methods used: {adaptive_results.methods_used}")
            print(f"Total results: {adaptive_results.total_unique_results}")
            
            # Test research-focused search
            research_context = {
                "keywords": ["pizza", "service", "quality", "restaurant"],
                "research_goals": ["identify_quality_factors", "understand_customer_preferences"]
            }
            
            research_results = await retriever.research_focused_search(
                query, 
                research_context=research_context,
                k=5
            )
            print(f"\nResearch-Focused Search Results:")
            print(f"Methods used: {research_results.methods_used}")
            print(f"Total results: {research_results.total_unique_results}")
    
    # Run the async function
    asyncio.run(main())