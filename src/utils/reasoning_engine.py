"""
Multi-Step Reasoning Engine following IterDRAG pattern for deep research
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..agents.manager_agent import ManagerAgent
from ..agents.research_agent import ResearchAgent
from ..retrieval.vector_store import VectorStoreManager
from ..embeddings.ollama_embeddings import OllamaEmbeddingService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningStage(Enum):
    """Enum for different reasoning stages"""
    INITIALIZATION = "initialization"
    QUERY_DECOMPOSITION = "query_decomposition"
    ITERATIVE_RETRIEVAL = "iterative_retrieval"
    CONTEXT_BUILDING = "context_building"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class ReasoningStep:
    """Data class representing a single reasoning step"""
    stage: ReasoningStage
    step_id: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


class MultiStepReasoningEngine:
    """
    Advanced reasoning engine implementing IterDRAG pattern for deep research.
    
    Key features:
    - Iterative query decomposition and refinement
    - Dynamic retrieval and context building
    - Multi-agent coordination
    - Progressive knowledge accumulation
    - Quality validation and verification
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        embedding_service: Optional[OllamaEmbeddingService] = None,
        max_iterations: int = 5,
        context_window_size: int = 8192
    ):
        """
        Initialize the reasoning engine.
        
        Args:
            vector_store_manager: Vector store for document retrieval
            embedding_service: Embedding service for query processing
            max_iterations: Maximum number of reasoning iterations
            context_window_size: Maximum context window size in tokens
        """
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.max_iterations = max_iterations
        self.context_window_size = context_window_size
        
        # Initialize agents
        self.manager_agent = ManagerAgent(
            vector_store_manager=self.vector_store_manager
        )
        self.research_agent = ResearchAgent(
            vector_store_manager=self.vector_store_manager
        )
        
        # Reasoning state
        self.current_session_id: Optional[str] = None
        self.reasoning_steps: List[ReasoningStep] = []
        self.accumulated_context: Dict[str, Any] = {}
        self.iteration_count: int = 0
        
        logger.info("Initialized MultiStepReasoningEngine")
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Process a query using multi-step reasoning.
        
        Args:
            query: Research query to process
            context: Optional additional context
            reasoning_depth: Depth of reasoning (1-5)
            
        Returns:
            Dictionary with comprehensive research results
        """
        try:
            # Initialize new reasoning session
            session_id = await self._initialize_session(query, context or {})
            
            logger.info(f"Starting multi-step reasoning for query: {query[:100]}...")
            
            # Stage 1: Query Analysis and Decomposition
            decomposition_result = await self._execute_stage(
                ReasoningStage.QUERY_DECOMPOSITION,
                {"query": query, "context": context, "depth": reasoning_depth}
            )
            
            if not decomposition_result["success"]:
                return decomposition_result
            
            # Stage 2: Iterative Retrieval and Analysis
            retrieval_result = await self._execute_iterative_retrieval(
                query, decomposition_result, reasoning_depth
            )
            
            if not retrieval_result["success"]:
                return retrieval_result
            
            # Stage 3: Context Building and Integration
            context_result = await self._execute_stage(
                ReasoningStage.CONTEXT_BUILDING,
                {
                    "query": query,
                    "decomposition": decomposition_result,
                    "retrieval": retrieval_result,
                    "accumulated_context": self.accumulated_context
                }
            )
            
            # Stage 4: Synthesis and Validation
            synthesis_result = await self._execute_stage(
                ReasoningStage.SYNTHESIS,
                {
                    "query": query,
                    "all_findings": {
                        "decomposition": decomposition_result,
                        "retrieval": retrieval_result,
                        "context": context_result
                    }
                }
            )
            
            # Stage 5: Final Validation
            validation_result = await self._execute_stage(
                ReasoningStage.VALIDATION,
                {
                    "query": query,
                    "synthesis": synthesis_result,
                    "reasoning_steps": self.reasoning_steps
                }
            )
            
            # Compile final results
            final_result = await self._compile_final_results(
                query, synthesis_result, validation_result
            )
            
            logger.info(f"Completed multi-step reasoning with {len(self.reasoning_steps)} steps")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-step reasoning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": self.current_session_id,
                "steps_completed": len(self.reasoning_steps),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _initialize_session(self, query: str, context: Dict[str, Any]) -> str:
        """Initialize a new reasoning session."""
        import uuid
        
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.reasoning_steps.clear()
        self.accumulated_context.clear()
        self.iteration_count = 0
        
        # Store initial context
        self.accumulated_context.update({
            "session_id": session_id,
            "original_query": query,
            "initial_context": context,
            "started_at": datetime.now().isoformat()
        })
        
        # Reset agent states
        self.manager_agent.reset_research_state()
        self.research_agent.reset_research_session()
        
        logger.debug(f"Initialized reasoning session: {session_id}")
        return session_id
    
    async def _execute_stage(
        self,
        stage: ReasoningStage,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific reasoning stage."""
        import uuid
        import time
        
        step_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.debug(f"Executing stage: {stage.value}")
        
        try:
            if stage == ReasoningStage.QUERY_DECOMPOSITION:
                result = await self._decompose_query(input_data)
            elif stage == ReasoningStage.CONTEXT_BUILDING:
                result = await self._build_context(input_data)
            elif stage == ReasoningStage.SYNTHESIS:
                result = await self._synthesize_findings(input_data)
            elif stage == ReasoningStage.VALIDATION:
                result = await self._validate_results(input_data)
            else:
                raise ValueError(f"Unknown reasoning stage: {stage}")
            
            duration = time.time() - start_time
            
            # Record reasoning step
            step = ReasoningStep(
                stage=stage,
                step_id=step_id,
                description=f"Executed {stage.value}",
                input_data=input_data,
                output_data=result,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                success=result.get("success", False)
            )
            
            self.reasoning_steps.append(step)
            
            # Update accumulated context
            self.accumulated_context[f"{stage.value}_result"] = result
            
            logger.debug(f"Completed stage {stage.value} in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            error_result = {
                "success": False,
                "error": str(e),
                "stage": stage.value
            }
            
            step = ReasoningStep(
                stage=stage,
                step_id=step_id,
                description=f"Failed {stage.value}",
                input_data=input_data,
                output_data=error_result,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )
            
            self.reasoning_steps.append(step)
            
            logger.error(f"Stage {stage.value} failed: {e}")
            return error_result
    
    async def _decompose_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the query into sub-questions and research tasks."""
        try:
            query = input_data["query"]
            context = input_data.get("context", {})
            depth = input_data.get("depth", 3)
            
            # Use manager agent to create research plan
            manager_result = await self.manager_agent.process({
                "query": query,
                "context": context,
                "action": "plan",
                "reasoning_depth": depth
            })
            
            if manager_result.get("success"):
                plan = manager_result.get("plan", {})
                
                # Extract sub-queries from the plan
                sub_queries = []
                tasks = plan.get("tasks", [])
                
                for task in tasks:
                    sub_queries.extend(task.get("search_terms", []))
                
                return {
                    "success": True,
                    "original_query": query,
                    "research_plan": plan,
                    "sub_queries": sub_queries,
                    "task_count": len(tasks),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return manager_result
                
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_iterative_retrieval(
        self,
        query: str,
        decomposition_result: Dict[str, Any],
        reasoning_depth: int
    ) -> Dict[str, Any]:
        """Execute iterative retrieval following IterDRAG pattern."""
        try:
            logger.debug("Starting iterative retrieval process")
            
            retrieval_results = []
            current_context = ""
            
            # Get sub-queries from decomposition
            sub_queries = decomposition_result.get("sub_queries", [query])
            
            # Iterative retrieval for each sub-query
            for i, sub_query in enumerate(sub_queries[:reasoning_depth]):
                self.iteration_count += 1
                
                logger.debug(f"Iteration {self.iteration_count}: {sub_query}")
                
                # Refine query based on accumulated context
                refined_query = await self._refine_query_with_context(
                    sub_query, current_context
                )
                
                # Retrieve documents for refined query
                retrieval_result = await self.research_agent.process({
                    "query": refined_query,
                    "action": "retrieve",
                    "context": {"iteration": i, "accumulated_context": current_context}
                })
                
                if retrieval_result.get("success"):
                    # Analyze retrieved documents
                    analysis_result = await self.research_agent.process({
                        "query": refined_query,
                        "action": "analyze",
                        "context": {
                            "documents": retrieval_result.get("documents", []),
                            "previous_context": current_context
                        }
                    })
                    
                    retrieval_results.append({
                        "iteration": i + 1,
                        "sub_query": sub_query,
                        "refined_query": refined_query,
                        "retrieval": retrieval_result,
                        "analysis": analysis_result
                    })
                    
                    # Update context for next iteration
                    if analysis_result.get("success"):
                        analysis_text = analysis_result.get("analysis_result", "")
                        current_context += f"\n\nIteration {i+1} findings:\n{analysis_text}"
                
                # Prevent infinite loops
                if self.iteration_count >= self.max_iterations:
                    logger.warning(f"Reached maximum iterations ({self.max_iterations})")
                    break
            
            return {
                "success": True,
                "retrieval_results": retrieval_results,
                "iterations_completed": self.iteration_count,
                "final_context": current_context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Iterative retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations_completed": self.iteration_count
            }
    
    async def _refine_query_with_context(self, query: str, context: str) -> str:
        """Refine a query based on accumulated context."""
        try:
            if not context.strip():
                return query
            
            # Use manager agent to refine the query
            refinement_prompt = f"""
Given the accumulated research context, refine this query for better information retrieval:

Original Query: {query}

Accumulated Context:
{context}

Provide a refined query that:
1. Builds upon what we already know
2. Focuses on missing information gaps
3. Uses more specific terminology where appropriate
4. Avoids redundancy with previous findings

Return only the refined query, nothing else.
"""
            
            refined_query = await self.manager_agent.generate_response(
                refinement_prompt, include_history=False
            )
            
            # Clean up the response to extract just the query
            refined_query = refined_query.strip().split('\n')[0]
            
            logger.debug(f"Refined query: {query} -> {refined_query}")
            return refined_query
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return query  # Fallback to original query
    
    async def _build_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context from all retrieval results."""
        try:
            retrieval_result = input_data.get("retrieval", {})
            retrieval_results = retrieval_result.get("retrieval_results", [])
            
            # Combine all findings
            combined_findings = []
            all_sources = []
            
            for result in retrieval_results:
                analysis = result.get("analysis", {})
                if analysis.get("success"):
                    combined_findings.append({
                        "iteration": result.get("iteration"),
                        "query": result.get("refined_query"),
                        "findings": analysis.get("analysis_result", "")
                    })
                
                retrieval = result.get("retrieval", {})
                documents = retrieval.get("documents", [])
                for doc in documents:
                    all_sources.append(doc.get("source", "unknown"))
            
            # Remove duplicate sources
            unique_sources = list(set(all_sources))
            
            return {
                "success": True,
                "combined_findings": combined_findings,
                "total_iterations": len(retrieval_results),
                "unique_sources": unique_sources,
                "source_count": len(unique_sources),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _synthesize_findings(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all findings into a comprehensive response."""
        try:
            query = input_data["query"]
            all_findings = input_data["all_findings"]
            
            # Use research agent to synthesize
            synthesis_result = await self.research_agent.process({
                "query": query,
                "action": "synthesize",
                "context": all_findings
            })
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Findings synthesis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_results(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the final results for quality and completeness."""
        try:
            query = input_data["query"]
            synthesis = input_data["synthesis"]
            
            if not synthesis.get("success"):
                return {
                    "success": False,
                    "validation_status": "failed",
                    "error": "Synthesis failed, cannot validate"
                }
            
            synthesis_text = synthesis.get("synthesis", "")
            
            # Use manager agent for validation
            validation_prompt = f"""
Validate this research result for quality and completeness:

Original Query: {query}

Research Result:
{synthesis_text}

Evaluate the result on:
1. COMPLETENESS: Does it fully address the query?
2. ACCURACY: Are claims well-supported?
3. CLARITY: Is it clearly written and well-structured?
4. COVERAGE: Are important aspects covered?
5. RELIABILITY: Are sources credible and relevant?

Provide a validation score (1-10) and specific recommendations for improvement if needed.
Format: SCORE: X/10
ASSESSMENT: [detailed assessment]
RECOMMENDATIONS: [specific recommendations or "None needed"]
"""
            
            validation_response = await self.manager_agent.generate_response(
                validation_prompt, include_history=False
            )
            
            # Extract validation score
            score = 7  # Default score
            try:
                score_line = [line for line in validation_response.split('\n') if 'SCORE:' in line.upper()]
                if score_line:
                    score_text = score_line[0].split(':')[1].strip()
                    score = int(score_text.split('/')[0])
            except:
                pass
            
            return {
                "success": True,
                "validation_score": score,
                "validation_response": validation_response,
                "validation_status": "passed" if score >= 7 else "needs_improvement",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Results validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_status": "error"
            }
    
    async def _compile_final_results(
        self,
        query: str,
        synthesis_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile the final comprehensive results."""
        try:
            # Calculate total processing time
            total_duration = sum(step.duration_seconds for step in self.reasoning_steps)
            
            # Count successful steps
            successful_steps = sum(1 for step in self.reasoning_steps if step.success)
            
            # Get final answer
            final_answer = synthesis_result.get("synthesis", "Unable to generate synthesis")
            
            # Compile metadata
            metadata = {
                "session_id": self.current_session_id,
                "total_steps": len(self.reasoning_steps),
                "successful_steps": successful_steps,
                "total_duration_seconds": total_duration,
                "iterations_completed": self.iteration_count,
                "validation_score": validation_result.get("validation_score", 0),
                "validation_status": validation_result.get("validation_status", "unknown"),
                "reasoning_steps": [
                    {
                        "stage": step.stage.value,
                        "duration": step.duration_seconds,
                        "success": step.success
                    }
                    for step in self.reasoning_steps
                ]
            }
            
            return {
                "success": True,
                "query": query,
                "answer": final_answer,
                "confidence_score": validation_result.get("validation_score", 0) / 10.0,
                "metadata": metadata,
                "synthesis_result": synthesis_result,
                "validation_result": validation_result,
                "reasoning_trace": self.reasoning_steps,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Final results compilation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "metadata": {
                    "session_id": self.current_session_id,
                    "steps_attempted": len(self.reasoning_steps)
                }
            }
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current reasoning session."""
        return {
            "session_id": self.current_session_id,
            "total_steps": len(self.reasoning_steps),
            "successful_steps": sum(1 for step in self.reasoning_steps if step.success),
            "failed_steps": sum(1 for step in self.reasoning_steps if not step.success),
            "total_duration": sum(step.duration_seconds for step in self.reasoning_steps),
            "iterations_completed": self.iteration_count,
            "stages_completed": list(set(step.stage.value for step in self.reasoning_steps)),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_engine(self):
        """Reset the reasoning engine for a new session."""
        self.current_session_id = None
        self.reasoning_steps.clear()
        self.accumulated_context.clear()
        self.iteration_count = 0
        self.manager_agent.reset_research_state()
        self.research_agent.reset_research_session()
        logger.info("Reset reasoning engine")