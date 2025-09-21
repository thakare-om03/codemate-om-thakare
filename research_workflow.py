"""
Research Workflow Orchestration using LangGraph
Implements advanced research workflows with planning, execution, validation, and iterative refinement
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from advanced_retrieval import AdvancedRetriever, FusedResults
from research_agent import ResearchFinding, ResearchResult, ResearchQuery


# Enhanced models for workflow orchestration
class ResearchStrategy(BaseModel):
    """Research strategy definition"""
    strategy_name: str = Field(description="Name of the research strategy")
    search_methods: List[str] = Field(description="Search methods to use")
    validation_level: str = Field(description="Level of validation required")
    iteration_limit: int = Field(description="Maximum iterations for this strategy")
    confidence_threshold: float = Field(description="Minimum confidence threshold")


class WorkflowStep(BaseModel):
    """Individual workflow step"""
    step_name: str = Field(description="Name of the workflow step")
    step_type: str = Field(description="Type of step: search, analyze, validate, synthesize")
    parameters: Dict[str, Any] = Field(description="Parameters for the step")
    dependencies: List[str] = Field(description="Dependencies on other steps")
    success_criteria: List[str] = Field(description="Criteria for step success")


class ResearchPlan(BaseModel):
    """Comprehensive research plan"""
    research_objective: str = Field(description="Main research objective")
    strategy: ResearchStrategy = Field(description="Research strategy to use")
    workflow_steps: List[WorkflowStep] = Field(description="Sequence of workflow steps")
    quality_gates: List[str] = Field(description="Quality gates and checkpoints")
    estimated_duration: int = Field(description="Estimated duration in minutes")


class QualityAssessment(BaseModel):
    """Quality assessment of research results"""
    completeness_score: float = Field(description="How complete is the research (0-1)")
    accuracy_score: float = Field(description="Estimated accuracy of findings (0-1)")
    relevance_score: float = Field(description="Relevance to original query (0-1)")
    source_diversity: float = Field(description="Diversity of sources used (0-1)")
    confidence_level: float = Field(description="Overall confidence level (0-1)")
    areas_for_improvement: List[str] = Field(description="Areas that need improvement")
    validation_status: str = Field(description="Overall validation status")


# Enhanced workflow state
class EnhancedResearchState(TypedDict):
    """Enhanced state for research workflow orchestration"""
    # Basic state
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    research_objective: str
    
    # Planning state
    research_plan: Optional[ResearchPlan]
    current_strategy: Optional[ResearchStrategy]
    workflow_steps: List[WorkflowStep]
    current_step_index: int
    
    # Execution state
    search_results: List[FusedResults]
    findings: List[ResearchFinding]
    intermediate_results: Dict[str, Any]
    
    # Quality and validation
    quality_assessment: Optional[QualityAssessment]
    validation_results: List[Dict[str, Any]]
    confidence_scores: List[float]
    
    # Iteration control
    iteration_count: int
    max_iterations: int
    strategy_changes: List[str]
    
    # Output
    final_result: Optional[ResearchResult]
    research_artifacts: Dict[str, Any]


class ResearchWorkflowOrchestrator:
    """Advanced research workflow orchestrator"""
    
    def __init__(self, 
                 embedding_system: AdvancedEmbeddingSystem, 
                 config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.embedding_system = embedding_system
        
        # Initialize components
        self.retriever = AdvancedRetriever(embedding_system, config)
        
        # Initialize models
        model_config = self.config.get_ollama_config()
        self.llm = ChatOllama(**model_config["llm"])
        self.reasoning_llm = ChatOllama(**model_config["reasoning"])
        self.summarization_llm = ChatOllama(**model_config["summarization"])
        
        # Initialize parsers
        self.plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        self.strategy_parser = PydanticOutputParser(pydantic_object=ResearchStrategy)
        self.quality_parser = PydanticOutputParser(pydantic_object=QualityAssessment)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Build workflow graph
        self.workflow_graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the enhanced research workflow graph"""
        workflow = StateGraph(EnhancedResearchState)
        
        # Core workflow nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("create_research_plan", self._create_research_plan_node)
        workflow.add_node("execute_search_strategy", self._execute_search_strategy_node)
        workflow.add_node("analyze_search_results", self._analyze_search_results_node)
        workflow.add_node("validate_findings", self._validate_findings_node)
        workflow.add_node("assess_quality", self._assess_quality_node)
        workflow.add_node("refine_strategy", self._refine_strategy_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("generate_final_output", self._generate_final_output_node)
        
        # Enhanced workflow edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "create_research_plan")
        workflow.add_edge("create_research_plan", "execute_search_strategy")
        workflow.add_edge("execute_search_strategy", "analyze_search_results")
        workflow.add_edge("analyze_search_results", "validate_findings")
        workflow.add_edge("validate_findings", "assess_quality")
        
        # Conditional routing based on quality assessment
        workflow.add_conditional_edges(
            "assess_quality",
            self._quality_checkpoint,
            {
                "continue_research": "refine_strategy",
                "synthesize": "synthesize_results",
                "insufficient_data": "execute_search_strategy"
            }
        )
        
        workflow.add_edge("refine_strategy", "execute_search_strategy")
        workflow.add_edge("synthesize_results", "generate_final_output")
        workflow.add_edge("generate_final_output", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _analyze_query_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Analyze the research query and determine research objective"""
        try:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research analysis expert. Analyze the given query and determine:
                1. The main research objective
                2. Key concepts and themes
                3. Required depth of research
                4. Potential challenges or complexities
                5. Expected types of evidence needed
                
                Provide a clear, actionable research objective based on the query."""),
                ("human", "Query to analyze: {query}")
            ])
            
            response = await self.reasoning_llm.ainvoke(
                analysis_prompt.format_messages(query=state["original_query"])
            )
            
            # Extract research objective (simplified - could use structured parsing)
            research_objective = response.content.split('\n')[0] if response.content else state["original_query"]
            
            self.logger.info(f"Query analyzed. Research objective: {research_objective}")
            
            return {
                "research_objective": research_objective,
                "messages": [AIMessage(content=f"Research objective identified: {research_objective}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in query analysis: {e}")
            return {
                "research_objective": state["original_query"],
                "messages": [AIMessage(content=f"Error in query analysis: {str(e)}")]
            }
    
    async def _create_research_plan_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Create a comprehensive research plan"""
        try:
            planning_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research planning expert. Create a comprehensive research plan.
                
                Consider:
                1. The complexity of the research objective
                2. Available search methods and strategies
                3. Quality gates and validation requirements
                4. Iterative refinement needs
                5. Time and resource constraints
                
                {format_instructions}"""),
                ("human", """
                Research Objective: {objective}
                Original Query: {query}
                """)
            ])
            
            response = await self.reasoning_llm.ainvoke(
                planning_prompt.format_messages(
                    objective=state["research_objective"],
                    query=state["original_query"],
                    format_instructions=self.plan_parser.get_format_instructions()
                )
            )
            
            research_plan = self.plan_parser.parse(response.content)
            
            self.logger.info(f"Research plan created with {len(research_plan.workflow_steps)} steps")
            
            return {
                "research_plan": research_plan,
                "current_strategy": research_plan.strategy,
                "workflow_steps": research_plan.workflow_steps,
                "current_step_index": 0,
                "max_iterations": research_plan.strategy.iteration_limit,
                "messages": [AIMessage(content=f"Research plan created with strategy: {research_plan.strategy.strategy_name}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in research planning: {e}")
            # Fallback plan
            fallback_strategy = ResearchStrategy(
                strategy_name="basic_search",
                search_methods=["semantic", "keyword"],
                validation_level="basic",
                iteration_limit=3,
                confidence_threshold=0.6
            )
            
            return {
                "current_strategy": fallback_strategy,
                "max_iterations": 3,
                "messages": [AIMessage(content=f"Using fallback research strategy due to planning error: {str(e)}")]
            }
    
    async def _execute_search_strategy_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Execute the current search strategy"""
        try:
            current_strategy = state.get("current_strategy")
            if not current_strategy:
                return {"messages": [AIMessage(content="No search strategy available")]}
            
            query = state["original_query"]
            search_methods = current_strategy.search_methods
            
            # Determine search parameters based on strategy
            k = min(self.config.MAX_DOCUMENTS_PER_QUERY, 15)
            
            # Execute search based on strategy
            if current_strategy.strategy_name == "comprehensive":
                results = await self.retriever.research_focused_search(
                    query=query,
                    research_context={
                        "keywords": [],  # Could extract from query
                        "research_goals": [state["research_objective"]]
                    },
                    k=k
                )
            elif current_strategy.strategy_name == "focused":
                results = await self.retriever.adaptive_search(
                    query=query,
                    context=state["research_objective"],
                    k=k
                )
            else:
                # Default hybrid search
                results = await self.retriever.hybrid_search(
                    query=query,
                    k=k,
                    methods=search_methods,
                    fusion_strategy="rrf"
                )
            
            # Store results
            current_results = state.get("search_results", [])
            current_results.append(results)
            
            self.logger.info(f"Search executed. Found {results.total_unique_results} unique results using methods: {results.methods_used}")
            
            return {
                "search_results": current_results,
                "messages": [AIMessage(content=f"Search completed. Found {results.total_unique_results} results using {', '.join(results.methods_used)}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in search execution: {e}")
            return {
                "messages": [AIMessage(content=f"Error in search execution: {str(e)}")]
            }
    
    async def _analyze_search_results_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Analyze and process search results"""
        try:
            search_results = state.get("search_results", [])
            if not search_results:
                return {"messages": [AIMessage(content="No search results to analyze")]}
            
            # Get the latest search results
            latest_results = search_results[-1]
            
            # Convert to research findings
            findings = []
            for i, doc in enumerate(latest_results.documents):
                finding = ResearchFinding(
                    content=doc.page_content,
                    source=doc.metadata.get("source", f"document_{i}"),
                    relevance_score=latest_results.scores[i] if i < len(latest_results.scores) else 0.5,
                    confidence_score=0.8,  # Could be calculated more sophisticatedly
                    supporting_evidence=[],
                    related_queries=[]
                )
                findings.append(finding)
            
            # Merge with existing findings
            existing_findings = state.get("findings", [])
            all_findings = existing_findings + findings
            
            # Remove duplicates based on content similarity
            unique_findings = self._remove_duplicate_findings(all_findings)
            
            self.logger.info(f"Analyzed search results. Total unique findings: {len(unique_findings)}")
            
            return {
                "findings": unique_findings,
                "messages": [AIMessage(content=f"Search results analyzed. {len(unique_findings)} unique findings identified")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in result analysis: {e}")
            return {
                "messages": [AIMessage(content=f"Error in result analysis: {str(e)}")]
            }
    
    async def _validate_findings_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Validate research findings"""
        try:
            findings = state.get("findings", [])
            if not findings:
                return {"messages": [AIMessage(content="No findings to validate")]}
            
            validation_results = []
            
            # Validate top findings
            top_findings = sorted(findings, key=lambda x: x.relevance_score, reverse=True)[:5]
            
            for finding in top_findings:
                # Simple validation based on source and content length
                validation_score = 0.5
                
                # Boost score for longer, more detailed content
                if len(finding.content) > 200:
                    validation_score += 0.2
                
                # Boost score for specific sources
                if finding.source and "unknown" not in finding.source.lower():
                    validation_score += 0.1
                
                # Boost score for high relevance
                if finding.relevance_score > 0.7:
                    validation_score += 0.2
                
                validation_result = {
                    "finding_id": f"finding_{top_findings.index(finding)}",
                    "validation_score": min(validation_score, 1.0),
                    "issues": [],
                    "recommendations": []
                }
                
                validation_results.append(validation_result)
            
            # Calculate overall validation score
            avg_validation_score = sum(r["validation_score"] for r in validation_results) / max(len(validation_results), 1)
            
            current_validation_results = state.get("validation_results", [])
            current_validation_results.extend(validation_results)
            
            self.logger.info(f"Validation completed. Average validation score: {avg_validation_score:.2f}")
            
            return {
                "validation_results": current_validation_results,
                "messages": [AIMessage(content=f"Findings validated. Average validation score: {avg_validation_score:.2f}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in validation: {e}")
            return {
                "messages": [AIMessage(content=f"Error in validation: {str(e)}")]
            }
    
    async def _assess_quality_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Assess the quality of current research"""
        try:
            findings = state.get("findings", [])
            validation_results = state.get("validation_results", [])
            
            # Calculate quality metrics
            completeness_score = min(len(findings) / 10, 1.0)  # Target 10 findings
            
            avg_validation_score = sum(r["validation_score"] for r in validation_results) / max(len(validation_results), 1) if validation_results else 0.5
            
            source_diversity = len(set(f.source for f in findings)) / max(len(findings), 1) if findings else 0.0
            
            relevance_score = sum(f.relevance_score for f in findings) / max(len(findings), 1) if findings else 0.0
            
            confidence_level = min(avg_validation_score * relevance_score, 1.0)
            
            # Identify areas for improvement
            areas_for_improvement = []
            if completeness_score < 0.7:
                areas_for_improvement.append("Need more comprehensive search")
            if source_diversity < 0.5:
                areas_for_improvement.append("Need more diverse sources")
            if relevance_score < 0.6:
                areas_for_improvement.append("Need more relevant results")
            
            # Determine validation status
            if confidence_level >= 0.8:
                validation_status = "high_confidence"
            elif confidence_level >= 0.6:
                validation_status = "moderate_confidence"
            else:
                validation_status = "low_confidence"
            
            quality_assessment = QualityAssessment(
                completeness_score=completeness_score,
                accuracy_score=avg_validation_score,
                relevance_score=relevance_score,
                source_diversity=source_diversity,
                confidence_level=confidence_level,
                areas_for_improvement=areas_for_improvement,
                validation_status=validation_status
            )
            
            self.logger.info(f"Quality assessment completed. Confidence level: {confidence_level:.2f}, Status: {validation_status}")
            
            return {
                "quality_assessment": quality_assessment,
                "messages": [AIMessage(content=f"Quality assessment: {validation_status} (confidence: {confidence_level:.2f})")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return {
                "messages": [AIMessage(content=f"Error in quality assessment: {str(e)}")]
            }
    
    def _quality_checkpoint(self, state: EnhancedResearchState) -> str:
        """Determine next step based on quality assessment"""
        quality_assessment = state.get("quality_assessment")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if not quality_assessment:
            return "insufficient_data"
        
        # Check if we've reached max iterations
        if iteration_count >= max_iterations:
            return "synthesize"
        
        # Check quality thresholds
        if quality_assessment.confidence_level >= 0.8:
            return "synthesize"
        elif quality_assessment.confidence_level >= 0.6:
            if len(quality_assessment.areas_for_improvement) <= 1:
                return "synthesize"
            else:
                return "continue_research"
        else:
            return "continue_research"
    
    async def _refine_strategy_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Refine the research strategy based on quality assessment"""
        try:
            quality_assessment = state.get("quality_assessment")
            current_strategy = state.get("current_strategy")
            
            if not quality_assessment or not current_strategy:
                return {"messages": [AIMessage(content="Cannot refine strategy without quality assessment")]}
            
            # Create refined strategy based on areas for improvement
            strategy_changes = []
            new_search_methods = current_strategy.search_methods.copy()
            
            for area in quality_assessment.areas_for_improvement:
                if "comprehensive" in area.lower():
                    if "multi_query" not in new_search_methods:
                        new_search_methods.append("multi_query")
                        strategy_changes.append("Added multi-query search for comprehensiveness")
                
                if "diverse" in area.lower():
                    if "mmr" not in new_search_methods:
                        new_search_methods.append("mmr")
                        strategy_changes.append("Added MMR search for source diversity")
                
                if "relevant" in area.lower():
                    if "compressed" not in new_search_methods:
                        new_search_methods.append("compressed")
                        strategy_changes.append("Added compressed search for relevance")
            
            # Create refined strategy
            refined_strategy = ResearchStrategy(
                strategy_name=f"refined_{current_strategy.strategy_name}",
                search_methods=new_search_methods,
                validation_level="enhanced",
                iteration_limit=current_strategy.iteration_limit,
                confidence_threshold=max(current_strategy.confidence_threshold - 0.1, 0.5)
            )
            
            self.logger.info(f"Strategy refined. Changes: {', '.join(strategy_changes)}")
            
            return {
                "current_strategy": refined_strategy,
                "strategy_changes": state.get("strategy_changes", []) + strategy_changes,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Strategy refined: {', '.join(strategy_changes)}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in strategy refinement: {e}")
            return {
                "messages": [AIMessage(content=f"Error in strategy refinement: {str(e)}")]
            }
    
    async def _synthesize_results_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Synthesize all research findings"""
        try:
            findings = state.get("findings", [])
            quality_assessment = state.get("quality_assessment")
            
            if not findings:
                return {"messages": [AIMessage(content="No findings to synthesize")]}
            
            # Create synthesis prompt
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research synthesizer. Create a comprehensive synthesis of the research findings.
                
                Provide:
                1. A clear, well-structured synthesis of all findings
                2. Key insights and patterns identified
                3. Supporting evidence for main conclusions
                4. Limitations and gaps in the research
                5. Overall assessment of the research quality
                
                Focus on creating a coherent narrative that addresses the original research objective."""),
                ("human", """
                Research Objective: {objective}
                Number of Findings: {num_findings}
                Quality Assessment: {quality}
                
                Key Findings:
                {findings_text}
                """)
            ])
            
            # Prepare findings text
            findings_text = "\n\n".join([
                f"Finding {i+1} (Relevance: {f.relevance_score:.2f}):\n{f.content[:300]}..."
                for i, f in enumerate(sorted(findings, key=lambda x: x.relevance_score, reverse=True)[:10])
            ])
            
            response = await self.summarization_llm.ainvoke(
                synthesis_prompt.format_messages(
                    objective=state["research_objective"],
                    num_findings=len(findings),
                    quality=quality_assessment.validation_status if quality_assessment else "unknown",
                    findings_text=findings_text
                )
            )
            
            synthesis_text = response.content
            
            self.logger.info("Research synthesis completed")
            
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "synthesis": synthesis_text
                },
                "messages": [AIMessage(content="Research synthesis completed")]
            }
            
        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            return {
                "messages": [AIMessage(content=f"Error in synthesis: {str(e)}")]
            }
    
    async def _generate_final_output_node(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Generate the final research output"""
        try:
            findings = state.get("findings", [])
            quality_assessment = state.get("quality_assessment")
            intermediate_results = state.get("intermediate_results", {})
            strategy_changes = state.get("strategy_changes", [])
            
            # Create research query object
            research_query = ResearchQuery(
                original_query=state["original_query"],
                subqueries=[state["research_objective"]],
                keywords=[],  # Could extract from findings
                research_goals=[state["research_objective"]],
                estimated_complexity="medium",
                required_sources=len(set(f.source for f in findings))
            )
            
            # Create final result
            final_result = ResearchResult(
                query=research_query,
                findings=findings,
                synthesis=intermediate_results.get("synthesis", "Synthesis not available"),
                conclusions=[],  # Could extract from synthesis
                confidence_score=quality_assessment.confidence_level if quality_assessment else 0.5,
                sources_used=list(set(f.source for f in findings)),
                reasoning_steps=[msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
            )
            
            # Create research artifacts
            research_artifacts = {
                "quality_assessment": quality_assessment.model_dump() if quality_assessment else {},
                "strategy_changes": strategy_changes,
                "search_methods_used": state.get("current_strategy").search_methods if state.get("current_strategy") else [],
                "iteration_count": state.get("iteration_count", 0),
                "total_findings": len(findings)
            }
            
            self.logger.info(f"Final research output generated. Confidence: {final_result.confidence_score:.2f}")
            
            return {
                "final_result": final_result,
                "research_artifacts": research_artifacts,
                "messages": [AIMessage(content=f"Research completed successfully. Confidence score: {final_result.confidence_score:.2f}")]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating final output: {e}")
            return {
                "messages": [AIMessage(content=f"Error generating final output: {str(e)}")]
            }
    
    def _remove_duplicate_findings(self, findings: List[ResearchFinding]) -> List[ResearchFinding]:
        """Remove duplicate findings based on content similarity"""
        if not findings:
            return findings
        
        unique_findings = []
        seen_content = set()
        
        for finding in findings:
            # Use first 150 characters as content signature
            content_signature = finding.content[:150].strip().lower()
            
            if content_signature not in seen_content:
                unique_findings.append(finding)
                seen_content.add(content_signature)
        
        return unique_findings
    
    async def orchestrate_research(self, query: str, config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Orchestrate the complete research workflow"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "research_objective": "",
                "current_step_index": 0,
                "search_results": [],
                "findings": [],
                "intermediate_results": {},
                "validation_results": [],
                "confidence_scores": [],
                "iteration_count": 0,
                "max_iterations": config_override.get("max_iterations", 3) if config_override else 3,
                "strategy_changes": [],
                "research_artifacts": {}
            }
            
            # Execute workflow with checkpointer configuration
            config = {
                "configurable": {
                    "thread_id": f"research_session_{hash(query) % 1000000}",
                    "checkpoint_ns": "research"
                }
            }
            final_state = await self.workflow_graph.ainvoke(initial_state, config=config)
            
            return {
                "status": "success",
                "result": final_state.get("final_result"),
                "artifacts": final_state.get("research_artifacts", {}),
                "messages": [msg.content for msg in final_state.get("messages", []) if isinstance(msg, AIMessage)]
            }
            
        except Exception as e:
            self.logger.error(f"Error in research orchestration: {e}")
            return {
                "status": "error",
                "error": str(e),
                "result": None,
                "artifacts": {}
            }


# Usage example
if __name__ == "__main__":
    async def main():
        from embedding_system import AdvancedEmbeddingSystem
        
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add documents
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Document loading: {result}")
        
        if result.get("status") == "success":
            # Create orchestrator
            orchestrator = ResearchWorkflowOrchestrator(embedding_system, config)
            
            # Run research
            research_result = await orchestrator.orchestrate_research(
                "What factors contribute to excellent customer service in restaurants based on customer feedback?"
            )
            
            print(f"\nOrchestration Status: {research_result['status']}")
            
            if research_result['status'] == 'success':
                result = research_result['result']
                artifacts = research_result['artifacts']
                
                print(f"\nResearch Results:")
                print(f"Confidence Score: {result.confidence_score:.2f}")
                print(f"Number of Findings: {len(result.findings)}")
                print(f"Sources Used: {len(result.sources_used)}")
                print(f"\nSynthesis Preview:")
                print(result.synthesis[:500] + "..." if len(result.synthesis) > 500 else result.synthesis)
                
                print(f"\nResearch Artifacts:")
                print(f"Iterations: {artifacts.get('iteration_count', 0)}")
                print(f"Strategy Changes: {len(artifacts.get('strategy_changes', []))}")
                print(f"Search Methods: {artifacts.get('search_methods_used', [])}")
    
    # Run the async function
    asyncio.run(main())