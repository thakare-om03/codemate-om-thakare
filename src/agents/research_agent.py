"""
Research Agent for document retrieval, analysis, and information synthesis
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.documents import Document

from .base_agent import BaseAgent
from ..retrieval.vector_store import VectorStoreManager
from ..embeddings.ollama_embeddings import OllamaEmbeddingService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research Agent responsible for:
    - Document retrieval from vector database
    - Multi-step reasoning and analysis
    - Information synthesis and contextualization
    - Source validation and quality assessment
    """
    
    def __init__(
        self,
        name: str = "ResearchAgent",
        model: str = "llama3.2",
        vector_store_manager: Optional[VectorStoreManager] = None,
        **kwargs
    ):
        system_prompt = """You are an expert Research Agent specialized in comprehensive information retrieval and analysis.

Your core capabilities include:
1. INTELLIGENT RETRIEVAL: Find relevant information using semantic search and contextual understanding
2. MULTI-STEP REASONING: Break down complex queries and build comprehensive answers iteratively
3. SOURCE ANALYSIS: Evaluate source quality, relevance, and reliability
4. SYNTHESIS: Combine information from multiple sources into coherent, well-structured responses
5. CONTEXTUALIZATION: Provide relevant context and background information

Research methodology:
- Use iterative retrieval to build comprehensive understanding
- Cross-reference multiple sources for accuracy
- Provide clear citations and source attribution
- Identify gaps in available information
- Maintain objectivity and balanced perspective

Quality standards:
- Prioritize authoritative and recent sources
- Clearly distinguish between facts and opinions
- Acknowledge limitations and uncertainties
- Provide evidence-based conclusions
- Structure responses logically and clearly

You have access to a comprehensive vector database of documents for information retrieval."""
        
        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Initialize vector store
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        
        # Research state
        self.current_query: Optional[str] = None
        self.retrieved_documents: List[Tuple[Document, float]] = []
        self.analysis_history: List[Dict[str, Any]] = []
        self.synthesis_cache: Dict[str, str] = {}
        
        # Research tools
        self._setup_research_tools()
        
    def _setup_research_tools(self):
        """Set up research-specific tools."""
        
        @tool
        def search_documents(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
            """
            Search for relevant documents in the vector database.
            
            Args:
                query: Search query
                n_results: Number of results to return
                
            Returns:
                List of relevant documents with metadata
            """
            try:
                results = self.vector_store_manager.similarity_search(
                    query=query,
                    n_results=n_results,
                    include_scores=True
                )
                
                # Convert to serializable format
                search_results = []
                for doc, score in results:
                    search_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": score
                    })
                
                logger.debug(f"Retrieved {len(search_results)} documents for query: {query[:50]}...")
                return search_results
                
            except Exception as e:
                logger.error(f"Document search failed: {e}")
                return []
        
        @tool
        def get_document_statistics() -> Dict[str, Any]:
            """
            Get statistics about the document collection.
            
            Returns:
                Dictionary with collection statistics
            """
            try:
                info = self.vector_store_manager.get_collection_info()
                return {
                    "total_documents": info.get("document_count", 0),
                    "collection_name": info.get("name", "unknown"),
                    "metadata": info.get("metadata", {})
                }
            except Exception as e:
                logger.error(f"Failed to get document statistics: {e}")
                return {"error": str(e)}
        
        # Add tools to the agent
        self.tools.extend([search_documents, get_document_statistics])
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research requests and return comprehensive analysis.
        
        Args:
            input_data: Contains query, action, and context
            
        Returns:
            Dictionary with research results
        """
        try:
            query = input_data.get("query", "")
            action = input_data.get("action", "research")
            context = input_data.get("context", {})
            
            logger.info(f"Research agent processing: {action} for query: {query[:100]}...")
            
            self.current_query = query
            
            if action == "research":
                return await self.conduct_research(query, context)
            elif action == "retrieve":
                return await self.retrieve_documents(query, context)
            elif action == "analyze":
                return await self.analyze_documents(query, context)
            elif action == "synthesize":
                return await self.synthesize_findings(query, context)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Research agent processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def conduct_research(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct comprehensive research on the given query.
        
        Args:
            query: Research question
            context: Additional context information
            
        Returns:
            Dictionary with comprehensive research results
        """
        try:
            logger.info(f"Conducting comprehensive research for: {query}")
            
            # Step 1: Initial document retrieval
            retrieval_result = await self.retrieve_documents(query, context)
            if not retrieval_result.get("success"):
                return retrieval_result
            
            documents = retrieval_result.get("documents", [])
            
            # Step 2: Iterative analysis and refinement
            analysis_result = await self.iterative_analysis(query, documents)
            
            # Step 3: Synthesize final response
            synthesis_result = await self.synthesize_findings(query, {
                "documents": documents,
                "analysis": analysis_result,
                "context": context
            })
            
            return {
                "success": True,
                "query": query,
                "research_result": synthesis_result.get("synthesis", ""),
                "sources_used": len(documents),
                "analysis_steps": len(self.analysis_history),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "retrieval_result": retrieval_result,
                    "analysis_result": analysis_result,
                    "synthesis_result": synthesis_result
                }
            }
            
        except Exception as e:
            logger.error(f"Research conduct failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def retrieve_documents(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant documents using multi-step retrieval strategy.
        
        Args:
            query: Search query
            context: Additional context for retrieval
            
        Returns:
            Dictionary with retrieved documents
        """
        try:
            logger.debug(f"Retrieving documents for query: {query}")
            
            # Primary retrieval
            primary_results = self.vector_store_manager.similarity_search(
                query=query,
                n_results=10,
                include_scores=True
            )
            
            # Store retrieved documents
            self.retrieved_documents = primary_results
            
            # Generate query variations for broader coverage
            query_variations = await self._generate_query_variations(query)
            
            # Secondary retrieval with variations
            secondary_results = []
            for variation in query_variations[:3]:  # Limit to top 3 variations
                variation_results = self.vector_store_manager.similarity_search(
                    query=variation,
                    n_results=5,
                    include_scores=True
                )
                secondary_results.extend(variation_results)
            
            # Combine and deduplicate results
            all_results = primary_results + secondary_results
            unique_results = self._deduplicate_documents(all_results)
            
            # Convert to serializable format
            documents = []
            for doc, score in unique_results[:15]:  # Limit to top 15 results
                documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                    "source": doc.metadata.get("filename", "unknown")
                })
            
            logger.info(f"Retrieved {len(documents)} unique documents")
            
            return {
                "success": True,
                "documents": documents,
                "total_retrieved": len(unique_results),
                "query_variations_used": len(query_variations),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_query_variations(self, original_query: str) -> List[str]:
        """Generate variations of the query for broader retrieval coverage."""
        try:
            variation_prompt = f"""
Generate 3-5 alternative phrasings or related queries for this research question:

Original Query: {original_query}

Provide variations that would help find related or complementary information. Consider:
- Synonyms and alternative terminology
- Different aspects or angles of the topic
- More specific or more general versions
- Related concepts or subtopics

Return only the alternative queries, one per line, without numbering or additional text.
"""
            
            response = await self.generate_response(variation_prompt, include_history=False)
            
            # Extract query variations from response
            variations = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '?' in line:
                    variations.append(line)
            
            logger.debug(f"Generated {len(variations)} query variations")
            return variations
            
        except Exception as e:
            logger.error(f"Failed to generate query variations: {e}")
            return []
    
    def _deduplicate_documents(self, documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        seen_content = set()
        
        for doc, score in documents:
            # Create a simplified version of content for comparison
            content_key = re.sub(r'\s+', ' ', doc.page_content[:200]).strip().lower()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append((doc, score))
        
        # Sort by relevance score (higher is better)
        unique_docs.sort(key=lambda x: x[1], reverse=True)
        
        return unique_docs
    
    async def iterative_analysis(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform iterative analysis building comprehensive understanding.
        
        Args:
            query: Research question
            documents: Retrieved documents
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.debug(f"Starting iterative analysis for {len(documents)} documents")
            
            analysis_steps = []
            current_understanding = ""
            
            # Group documents by relevance
            high_relevance = [doc for doc in documents if doc.get("relevance_score", 0) > 0.8]
            medium_relevance = [doc for doc in documents if 0.6 <= doc.get("relevance_score", 0) <= 0.8]
            
            # Step 1: Analyze high-relevance documents
            if high_relevance:
                step1_analysis = await self._analyze_document_group(
                    query, high_relevance, "high-relevance sources"
                )
                analysis_steps.append(step1_analysis)
                current_understanding = step1_analysis.get("analysis", "")
            
            # Step 2: Analyze medium-relevance documents with context
            if medium_relevance and current_understanding:
                step2_analysis = await self._analyze_document_group(
                    query, medium_relevance, "additional sources", current_understanding
                )
                analysis_steps.append(step2_analysis)
                current_understanding += "\n\n" + step2_analysis.get("analysis", "")
            
            # Step 3: Cross-reference and validate findings
            if len(analysis_steps) > 1:
                validation_analysis = await self._validate_findings(query, analysis_steps)
                analysis_steps.append(validation_analysis)
            
            self.analysis_history.extend(analysis_steps)
            
            return {
                "success": True,
                "analysis_steps": analysis_steps,
                "final_understanding": current_understanding,
                "documents_analyzed": len(documents),
                "high_relevance_count": len(high_relevance),
                "medium_relevance_count": len(medium_relevance)
            }
            
        except Exception as e:
            logger.error(f"Iterative analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_document_group(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        group_name: str,
        prior_context: str = ""
    ) -> Dict[str, Any]:
        """Analyze a group of documents."""
        try:
            # Prepare documents text
            docs_text = ""
            for i, doc in enumerate(documents):
                docs_text += f"\n--- Document {i+1} (Score: {doc.get('relevance_score', 0):.2f}) ---\n"
                docs_text += doc.get("content", "")
                docs_text += f"\nSource: {doc.get('source', 'unknown')}\n"
            
            context_section = f"\nPrior Analysis Context:\n{prior_context}\n" if prior_context else ""
            
            analysis_prompt = f"""
Analyze these {group_name} to answer the research question:

Research Question: {query}
{context_section}
Documents to Analyze:
{docs_text}

Provide a structured analysis covering:

1. KEY FINDINGS: Main insights relevant to the research question
2. EVIDENCE: Supporting evidence and data from the documents
3. PATTERNS: Common themes or patterns across sources
4. GAPS: Information gaps or areas needing further research
5. RELIABILITY: Assessment of source quality and reliability

Focus on information directly relevant to answering the research question.
Be precise and cite specific sources when making claims.
"""
            
            analysis_response = await self.generate_response(analysis_prompt, include_history=False)
            
            return {
                "group_name": group_name,
                "documents_count": len(documents),
                "analysis": analysis_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document group analysis failed: {e}")
            return {
                "group_name": group_name,
                "error": str(e)
            }
    
    async def _validate_findings(self, query: str, analysis_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-reference and validate findings from multiple analysis steps."""
        try:
            # Combine all analysis results
            combined_analysis = ""
            for step in analysis_steps:
                combined_analysis += f"\n--- {step.get('group_name', 'Analysis')} ---\n"
                combined_analysis += step.get("analysis", "")
                combined_analysis += "\n"
            
            validation_prompt = f"""
Cross-reference and validate these research findings for the query: {query}

Analysis Results to Validate:
{combined_analysis}

Provide a validation assessment covering:

1. CONSISTENCY: Are the findings consistent across sources?
2. CONTRADICTIONS: Identify any contradictory information
3. CONFIDENCE: Assess confidence level in the findings
4. LIMITATIONS: Acknowledge limitations and uncertainties
5. SYNTHESIS: Integrate validated findings into coherent conclusions

Focus on providing a balanced, objective assessment of the research findings.
"""
            
            validation_response = await self.generate_response(validation_prompt, include_history=False)
            
            return {
                "group_name": "validation",
                "analysis": validation_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Findings validation failed: {e}")
            return {
                "group_name": "validation",
                "error": str(e)
            }
    
    async def synthesize_findings(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize all findings into a comprehensive response.
        
        Args:
            query: Original research question
            context: Context including documents and analysis
            
        Returns:
            Dictionary with synthesized findings
        """
        try:
            logger.debug("Synthesizing research findings")
            
            # Gather all available information
            documents = context.get("documents", [])
            analysis = context.get("analysis", {})
            
            # Prepare synthesis context
            synthesis_context = f"Research Question: {query}\n\n"
            
            # Add document summaries
            if documents:
                synthesis_context += "Key Sources:\n"
                for i, doc in enumerate(documents[:10]):  # Top 10 sources
                    synthesis_context += f"- Source {i+1}: {doc.get('source', 'unknown')} (Relevance: {doc.get('relevance_score', 0):.2f})\n"
                synthesis_context += "\n"
            
            # Add analysis findings
            if analysis.get("analysis_steps"):
                synthesis_context += "Analysis Findings:\n"
                for step in analysis["analysis_steps"]:
                    synthesis_context += f"- {step.get('group_name', 'Analysis')}: {step.get('analysis', '')[:200]}...\n"
                synthesis_context += "\n"
            
            synthesis_prompt = f"""
Based on the comprehensive research conducted, provide a definitive answer to this question:

{synthesis_context}

Create a comprehensive response that:

1. DIRECTLY ANSWERS the research question
2. PROVIDES EVIDENCE from the sources analyzed
3. ACKNOWLEDGES limitations or uncertainties
4. OFFERS additional context and implications
5. SUGGESTS areas for further research if relevant

Structure your response clearly and ensure it directly addresses what was asked.
Be authoritative where evidence supports conclusions, and cautious where evidence is limited.
"""
            
            synthesis_response = await self.generate_response(synthesis_prompt, include_history=False)
            
            # Cache the synthesis
            self.synthesis_cache[query] = synthesis_response
            
            return {
                "success": True,
                "synthesis": synthesis_response,
                "sources_used": len(documents),
                "word_count": len(synthesis_response.split()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Findings synthesis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get statistics about current research session."""
        return {
            "current_query": self.current_query,
            "documents_retrieved": len(self.retrieved_documents),
            "analysis_steps_completed": len(self.analysis_history),
            "synthesis_cache_size": len(self.synthesis_cache),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_research_session(self):
        """Reset the research session for a new query."""
        self.current_query = None
        self.retrieved_documents.clear()
        self.analysis_history.clear()
        self.synthesis_cache.clear()
        self.clear_history()
        logger.info("Reset research session")