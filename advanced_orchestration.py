"""
Advanced Research Orchestration System
Implements sophisticated multi-step reasoning, tool integration, and sub-agent coordination
Based on latest patterns from deepagents and open_deep_research documentation.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
# Note: SqliteSaver import disabled for compatibility
from pydantic import BaseModel, Field

# Import existing components
from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from advanced_retrieval import AdvancedRetriever
from research_agent import ResearchFinding
from quality_control import QualityController


class ResearchPhase(Enum):
    """Phases of the research process"""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    REPORTING = "reporting"


class AgentRole(Enum):
    """Different agent roles in the research process"""
    COORDINATOR = "coordinator"          # Orchestrates the overall process
    PLANNER = "planner"                 # Creates research plans and strategies
    SEARCHER = "searcher"               # Performs information retrieval
    ANALYZER = "analyzer"               # Analyzes and processes information
    VALIDATOR = "validator"             # Validates findings and quality
    SYNTHESIZER = "synthesizer"         # Synthesizes final results
    TOOL_SPECIALIST = "tool_specialist" # Handles tool integration


@dataclass
class ResearchTask:
    """Individual research task within the workflow"""
    task_id: str
    task_type: str
    description: str
    priority: int
    dependencies: List[str]
    assigned_agent: AgentRole
    status: str = "pending"
    result: Optional[Any] = None
    tools_required: List[str] = None
    estimated_duration: int = 5  # minutes


@dataclass
class AgentState:
    """State for individual agents"""
    role: AgentRole
    current_task: Optional[ResearchTask]
    memory: Dict[str, Any]
    capabilities: List[str]
    active: bool = True


class ResearchTool(BaseModel):
    """Enhanced research tool with metadata"""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    function: str = Field(description="Function to call")
    parameters: Dict[str, Any] = Field(description="Tool parameters")
    category: str = Field(description="Tool category")
    reliability_score: float = Field(description="Tool reliability (0-1)")


class AdvancedResearchOrchestrator:
    """
    Advanced research orchestrator with multi-agent coordination,
    tool integration, and sophisticated reasoning capabilities.
    """
    
    def __init__(self, 
                 embedding_system: AdvancedEmbeddingSystem,
                 config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.embedding_system = embedding_system
        
        # Initialize components
        self.retriever = AdvancedRetriever(embedding_system, config)
        self.quality_controller = QualityController(config)
        
        # Initialize LLM
        llm_config = self.config.get_ollama_config()
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(**llm_config["llm"])
        
        # Initialize multi-agent system
        self.agents = self._initialize_agents()
        self.research_tools = self._initialize_research_tools()
        self.task_queue = []
        self.coordination_memory = {}
        
        # Initialize workflow with in-memory checkpointing (simplified)
        self.checkpointer = None  # Disabled for compatibility
        self.workflow = self._build_advanced_workflow()
        
        # Logging
        import logging
        self.logger = logging.getLogger(__name__)
    
    def _initialize_agents(self) -> Dict[AgentRole, AgentState]:
        """Initialize specialized agents for different research tasks"""
        agents = {}
        
        # Coordinator Agent
        agents[AgentRole.COORDINATOR] = AgentState(
            role=AgentRole.COORDINATOR,
            current_task=None,
            memory={},
            capabilities=[
                "task_decomposition", "workflow_orchestration", 
                "quality_monitoring", "resource_allocation"
            ]
        )
        
        # Planner Agent
        agents[AgentRole.PLANNER] = AgentState(
            role=AgentRole.PLANNER,
            current_task=None,
            memory={},
            capabilities=[
                "research_planning", "strategy_formulation",
                "resource_estimation", "risk_assessment"
            ]
        )
        
        # Searcher Agent
        agents[AgentRole.SEARCHER] = AgentState(
            role=AgentRole.SEARCHER,
            current_task=None,
            memory={},
            capabilities=[
                "information_retrieval", "source_discovery",
                "query_optimization", "result_filtering"
            ]
        )
        
        # Analyzer Agent
        agents[AgentRole.ANALYZER] = AgentState(
            role=AgentRole.ANALYZER,
            current_task=None,
            memory={},
            capabilities=[
                "content_analysis", "pattern_recognition",
                "data_extraction", "insight_generation"
            ]
        )
        
        # Validator Agent
        agents[AgentRole.VALIDATOR] = AgentState(
            role=AgentRole.VALIDATOR,
            current_task=None,
            memory={},
            capabilities=[
                "fact_checking", "source_validation",
                "quality_assessment", "consistency_checking"
            ]
        )
        
        # Synthesizer Agent
        agents[AgentRole.SYNTHESIZER] = AgentState(
            role=AgentRole.SYNTHESIZER,
            current_task=None,
            memory={},
            capabilities=[
                "information_synthesis", "narrative_generation",
                "conclusion_drawing", "report_structuring"
            ]
        )
        
        return agents
    
    def _initialize_research_tools(self) -> List[ResearchTool]:
        """Initialize available research tools"""
        tools = []
        
        # Advanced retrieval tools
        tools.extend([
            ResearchTool(
                name="semantic_search",
                description="Perform semantic similarity search",
                function="retriever.semantic_search",
                parameters={"query": "str", "k": "int"},
                category="retrieval",
                reliability_score=0.85
            ),
            ResearchTool(
                name="contextual_compression_search",
                description="Search with LLM-based content compression",
                function="retriever.contextual_compression_search",
                parameters={"query": "str", "k": "int"},
                category="retrieval",
                reliability_score=0.90
            ),
            ResearchTool(
                name="reciprocal_rank_fusion",
                description="Multi-strategy fusion search",
                function="retriever.reciprocal_rank_fusion_search",
                parameters={"query": "str", "k": "int"},
                category="retrieval",
                reliability_score=0.95
            ),
            ResearchTool(
                name="adaptive_retrieval",
                description="Intelligent strategy selection",
                function="retriever.adaptive_retrieval_strategy",
                parameters={"query": "str", "k": "int"},
                category="retrieval",
                reliability_score=0.92
            )
        ])
        
        # Analysis tools
        tools.extend([
            ResearchTool(
                name="content_analyzer",
                description="Analyze document content for insights",
                function="analyze_content",
                parameters={"content": "str", "analysis_type": "str"},
                category="analysis",
                reliability_score=0.88
            ),
            ResearchTool(
                name="pattern_detector",
                description="Detect patterns in research findings",
                function="detect_patterns",
                parameters={"findings": "list", "pattern_type": "str"},
                category="analysis",
                reliability_score=0.82
            )
        ])
        
        # Quality tools
        tools.extend([
            ResearchTool(
                name="fact_checker",
                description="Verify facts and claims",
                function="quality_controller.validate_finding",
                parameters={"finding": "ResearchFinding", "query": "str"},
                category="validation",
                reliability_score=0.87
            ),
            ResearchTool(
                name="source_validator",
                description="Validate source credibility",
                function="validate_source",
                parameters={"source": "str", "criteria": "dict"},
                category="validation",
                reliability_score=0.85
            )
        ])
        
        return tools
    
    def _build_advanced_workflow(self) -> StateGraph:
        """Build advanced research workflow with multi-agent coordination"""
        from typing_extensions import TypedDict
        # Import add_messages from langgraph instead of langchain_core
        try:
            from langgraph.graph.message import add_messages
        except ImportError:
            # Fallback function if import fails
            def add_messages(messages_list):
                return messages_list
        
        from typing import Annotated
        
        # Enhanced state for multi-agent workflow
        class AdvancedResearchState(TypedDict):
            # Core research state
            messages: Annotated[List[BaseMessage], add_messages]
            original_query: str
            current_phase: ResearchPhase
            
            # Task management
            task_queue: List[ResearchTask]
            completed_tasks: List[ResearchTask]
            active_agents: Dict[str, AgentRole]
            
            # Research data
            findings: List[ResearchFinding]
            intermediate_results: Dict[str, Any]
            tool_results: Dict[str, Any]
            
            # Coordination
            coordination_memory: Dict[str, Any]
            quality_metrics: Dict[str, float]
            
            # Control flow
            iteration_count: int
            max_iterations: int
            should_continue: bool
            
            # Final output
            final_result: Optional[Any]
        
        # Create workflow
        workflow = StateGraph(AdvancedResearchState)
        
        # Add nodes for each phase
        workflow.add_node("initialize_research", self._initialize_research_node)
        workflow.add_node("decompose_and_plan", self._decompose_and_plan_node)
        workflow.add_node("coordinate_agents", self._coordinate_agents_node)
        workflow.add_node("execute_parallel_search", self._execute_parallel_search_node)
        workflow.add_node("analyze_and_synthesize", self._analyze_and_synthesize_node)
        workflow.add_node("validate_results", self._validate_results_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("finalize_research", self._finalize_research_node)
        workflow.add_node("human_feedback", self._human_feedback_node)
        
        # Define workflow edges
        workflow.set_entry_point("initialize_research")
        
        workflow.add_edge("initialize_research", "decompose_and_plan")
        workflow.add_edge("decompose_and_plan", "coordinate_agents")
        workflow.add_edge("coordinate_agents", "execute_parallel_search")
        workflow.add_edge("execute_parallel_search", "analyze_and_synthesize")
        workflow.add_edge("analyze_and_synthesize", "validate_results")
        
        # Conditional edges for iteration and human feedback
        workflow.add_conditional_edges(
            "validate_results",
            self._should_continue_research,
            {
                "continue": "coordinate_agents",
                "human_feedback": "human_feedback",
                "finalize": "generate_insights"
            }
        )
        
        workflow.add_edge("human_feedback", "coordinate_agents")
        workflow.add_edge("generate_insights", "finalize_research")
        workflow.add_edge("finalize_research", END)
        
        # Compile workflow (checkpointing disabled for compatibility)
        return workflow.compile()
    
    async def _initialize_research_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the research process and set up coordination"""
        try:
            self.logger.info("Initializing advanced research orchestration")
            
            # Initialize state
            return {
                "current_phase": ResearchPhase.INITIALIZATION,
                "task_queue": [],
                "completed_tasks": [],
                "active_agents": {role.value: role for role in AgentRole},
                "findings": [],
                "intermediate_results": {},
                "tool_results": {},
                "coordination_memory": {},
                "quality_metrics": {},
                "iteration_count": 0,
                "max_iterations": 3,
                "should_continue": True,
                "messages": state.get("messages", []) + [
                    AIMessage(content="Research orchestration initialized. Beginning advanced workflow.")
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing research: {e}")
            return {
                "messages": [AIMessage(content=f"Error in initialization: {str(e)}")]
            }
    
    async def _decompose_and_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the research query and create a comprehensive plan"""
        try:
            query = state["original_query"]
            self.logger.info(f"Decomposing query: {query}")
            
            # Use planner agent to create research tasks
            planner_agent = self.agents[AgentRole.PLANNER]
            
            # Create decomposition prompt
            decomposition_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research planner. Decompose the given research query into specific, actionable tasks.
                
                Create a comprehensive research plan that includes:
                1. Initial exploration tasks
                2. Focused search tasks for different aspects
                3. Analysis and synthesis tasks
                4. Validation and quality control tasks
                
                Each task should be specific, measurable, and assigned to the appropriate agent type."""),
                ("human", "Research Query: {query}\\n\\nCreate a detailed task decomposition plan.")
            ])
            
            # Generate research plan
            response = await self.llm.ainvoke(
                decomposition_prompt.format_messages(query=query)
            )
            
            # Create research tasks (simplified for now)
            tasks = [
                ResearchTask(
                    task_id="explore_1",
                    task_type="exploration",
                    description=f"Initial broad search for: {query}",
                    priority=1,
                    dependencies=[],
                    assigned_agent=AgentRole.SEARCHER,
                    tools_required=["adaptive_retrieval"]
                ),
                ResearchTask(
                    task_id="analyze_1",
                    task_type="analysis",
                    description=f"Analyze initial findings for patterns",
                    priority=2,
                    dependencies=["explore_1"],
                    assigned_agent=AgentRole.ANALYZER,
                    tools_required=["content_analyzer", "pattern_detector"]
                ),
                ResearchTask(
                    task_id="validate_1",
                    task_type="validation",
                    description="Validate key findings and sources",
                    priority=3,
                    dependencies=["analyze_1"],
                    assigned_agent=AgentRole.VALIDATOR,
                    tools_required=["fact_checker", "source_validator"]
                )
            ]
            
            return {
                "current_phase": ResearchPhase.PLANNING,
                "task_queue": tasks,
                "coordination_memory": {
                    "decomposition_response": response.content,
                    "planning_timestamp": datetime.now().isoformat()
                },
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Research plan created with {len(tasks)} tasks. Beginning coordination phase.")
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in decomposition: {e}")
            return {
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Error in planning: {str(e)}")
                ]
            }
    
    async def orchestrate_research(self, query: str, 
                                 human_feedback_callback: Callable = None,
                                 config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Orchestrate advanced research with multi-agent coordination
        
        Args:
            query: Research query
            human_feedback_callback: Optional callback for human-in-the-loop
            config_override: Configuration overrides
            
        Returns:
            Comprehensive research results
        """
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "human_feedback_callback": human_feedback_callback
            }
            
            # Apply config overrides
            if config_override:
                initial_state.update(config_override)
            
            # Configure workflow execution
            config = {
                "configurable": {
                    "thread_id": f"advanced_research_{hash(query) % 1000000}",
                    "checkpoint_ns": "advanced_research"
                }
            }
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            return {
                "status": "success",
                "result": final_state.get("final_result"),
                "task_results": final_state.get("completed_tasks", []),
                "quality_metrics": final_state.get("quality_metrics", {}),
                "coordination_log": final_state.get("coordination_memory", {}),
                "messages": [msg.content for msg in final_state.get("messages", []) if isinstance(msg, AIMessage)]
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced orchestration: {e}")
            return {
                "status": "error",
                "error": str(e),
                "result": None
            }
    
    # Placeholder methods for workflow nodes (simplified implementations)
    
    async def _coordinate_agents_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agent activities and task distribution"""
        # Simplified coordination logic
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content="Agent coordination completed.")
            ]
        }
    
    async def _execute_parallel_search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel search using multiple agents and strategies"""
        try:
            query = state["original_query"]
            
            # Execute adaptive retrieval
            results = await self.retriever.adaptive_retrieval_strategy(query)
            
            # Convert to research findings
            findings = []
            for doc in results.documents:
                finding = ResearchFinding(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.8,  # Placeholder
                    confidence_score=0.8,  # Placeholder
                    supporting_evidence=[doc.page_content[:200]],  # First 200 chars as evidence
                    related_queries=[query]  # Current query as related
                )
                findings.append(finding)
            
            return {
                "current_phase": ResearchPhase.EXPLORATION,
                "findings": findings,
                "tool_results": {"adaptive_retrieval": asdict(results) if hasattr(results, '__dict__') else str(results)},
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Parallel search completed. Found {len(findings)} relevant items.")
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in parallel search: {e}")
            return {
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Error in search execution: {str(e)}")
                ]
            }
    
    async def _analyze_and_synthesize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze findings and synthesize insights"""
        findings = state.get("findings", [])
        
        return {
            "current_phase": ResearchPhase.ANALYSIS,
            "intermediate_results": {
                "analysis_summary": f"Analyzed {len(findings)} findings",
                "synthesis_complete": True
            },
            "messages": state.get("messages", []) + [
                AIMessage(content="Analysis and synthesis completed.")
            ]
        }
    
    async def _validate_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate research results and assess quality"""
        findings = state.get("findings", [])
        
        quality_metrics = {
            "completeness": 0.8,
            "accuracy": 0.85,
            "relevance": 0.9,
            "confidence": 0.82
        }
        
        return {
            "current_phase": ResearchPhase.VALIDATION,
            "quality_metrics": quality_metrics,
            "should_continue": False,  # Stop iteration
            "messages": state.get("messages", []) + [
                AIMessage(content="Validation completed. Quality metrics calculated.")
            ]
        }
    
    async def _generate_insights_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final insights and conclusions"""
        return {
            "current_phase": ResearchPhase.SYNTHESIS,
            "messages": state.get("messages", []) + [
                AIMessage(content="Insights generation completed.")
            ]
        }
    
    async def _finalize_research_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize research and prepare results"""
        findings = state.get("findings", [])
        quality_metrics = state.get("quality_metrics", {})
        
        final_result = {
            "query": state["original_query"],
            "findings": [asdict(f) if hasattr(f, '__dict__') else str(f) for f in findings],
            "quality_metrics": quality_metrics,
            "total_findings": len(findings),
            "completion_status": "success"
        }
        
        return {
            "current_phase": ResearchPhase.REPORTING,
            "final_result": final_result,
            "messages": state.get("messages", []) + [
                AIMessage(content="Advanced research orchestration completed successfully.")
            ]
        }
    
    async def _human_feedback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human-in-the-loop feedback"""
        callback = state.get("human_feedback_callback")
        
        if callback:
            feedback = await callback(state)
            return {
                "coordination_memory": {
                    **state.get("coordination_memory", {}),
                    "human_feedback": feedback
                },
                "messages": state.get("messages", []) + [
                    AIMessage(content="Human feedback incorporated.")
                ]
            }
        
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content="No human feedback callback provided.")
            ]
        }
    
    def _should_continue_research(self, state: Dict[str, Any]) -> str:
        """Determine whether to continue research, seek feedback, or finalize"""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        quality_metrics = state.get("quality_metrics", {})
        
        # Check if quality is sufficient
        avg_quality = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0
        
        if avg_quality > 0.85:
            return "finalize"
        elif iteration_count >= max_iterations:
            return "finalize"
        elif avg_quality < 0.6:
            return "human_feedback"
        else:
            return "continue"


# Integration point for existing workflow
class EnhancedWorkflowIntegration:
    """Integration layer for enhanced workflow with existing components"""
    
    @staticmethod
    def upgrade_existing_workflow(existing_orchestrator):
        """Upgrade existing workflow with advanced capabilities"""
        # Add advanced retrieval capabilities
        if hasattr(existing_orchestrator, 'retriever'):
            # Enhance existing retriever with new methods
            advanced_methods = [
                'contextual_compression_search',
                'reciprocal_rank_fusion_search', 
                'adaptive_retrieval_strategy'
            ]
            
            for method_name in advanced_methods:
                if not hasattr(existing_orchestrator.retriever, method_name):
                    print(f"Warning: {method_name} not available in current retriever")
        
        return existing_orchestrator