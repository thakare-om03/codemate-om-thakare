"""
Manager Agent for query planning, task decomposition, and result synthesis
"""
import json
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..retrieval.vector_store import VectorStoreManager

logger = get_logger(__name__)


class ResearchTask(BaseModel):
    """Model for individual research tasks"""
    task_id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Description of what needs to be researched")
    priority: int = Field(description="Priority level (1=highest, 5=lowest)")
    dependencies: List[str] = Field(default=[], description="List of task IDs that must be completed first")
    expected_sources: int = Field(default=5, description="Expected number of sources to find")
    search_terms: List[str] = Field(description="List of search terms for this task")


class ResearchPlan(BaseModel):
    """Model for the complete research plan"""
    query: str = Field(description="Original research query")
    objective: str = Field(description="Overall research objective")
    tasks: List[ResearchTask] = Field(description="List of research tasks")
    estimated_duration: int = Field(description="Estimated duration in minutes")
    success_criteria: List[str] = Field(description="Criteria for successful completion")


class ManagerAgent(BaseAgent):
    """
    Manager Agent responsible for:
    - Query analysis and decomposition
    - Research task planning
    - Agent coordination
    - Result synthesis and quality control
    """
    
    def __init__(
        self,
        name: str = "ManagerAgent",
        model: str = "llama3.2",
        vector_store_manager: Optional['VectorStoreManager'] = None,
        **kwargs
    ):
        system_prompt = """You are a highly skilled Research Manager responsible for planning and coordinating complex research tasks.

Your responsibilities include:
1. QUERY ANALYSIS: Break down complex research questions into manageable sub-tasks
2. TASK PLANNING: Create detailed research plans with clear objectives and priorities
3. AGENT COORDINATION: Assign tasks to specialized research agents
4. QUALITY CONTROL: Ensure research quality and completeness
5. SYNTHESIS: Combine findings into comprehensive, well-structured reports

Key principles:
- Always think step-by-step and plan thoroughly before acting
- Decompose complex queries into 3-7 focused sub-tasks
- Prioritize tasks based on importance and dependencies
- Ensure each task has clear success criteria
- Focus on finding authoritative, relevant sources
- Synthesize findings into coherent, well-structured responses

You have access to research agents and vector database tools to gather information.
Always provide detailed reasoning for your decisions and maintain high research standards."""
        
        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Store vector store manager (if provided)
        self.vector_store_manager = vector_store_manager
        
        # Initialize output parsers
        self.plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        # Research state
        self.current_plan: Optional[ResearchPlan] = None
        self.task_results: Dict[str, Dict[str, Any]] = {}
        self.research_context: Dict[str, Any] = {}
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and coordinate research activities.
        
        Args:
            input_data: Contains query, context, and processing instructions
            
        Returns:
            Dictionary with processed results
        """
        try:
            query = input_data.get("query", "")
            context = input_data.get("context", {})
            action = input_data.get("action", "research")
            
            logger.info(f"Manager processing query: {query[:100]}...")
            
            if action == "plan":
                return await self.create_research_plan(query, context)
            elif action == "research":
                return await self.execute_research(query, context)
            elif action == "synthesize":
                return await self.synthesize_results(input_data.get("results", []))
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Manager agent processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def create_research_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed research plan for the given query.
        
        Args:
            query: Research question
            context: Additional context information
            
        Returns:
            Dictionary containing the research plan
        """
        try:
            logger.info(f"Creating research plan for query: {query}")
            
            # Analyze the query and create a research plan
            planning_prompt = f"""
Analyze this research query and create a detailed research plan:

QUERY: {query}

CONTEXT: {json.dumps(context, indent=2) if context else 'None provided'}

Create a comprehensive research plan that breaks down this query into 3-7 focused research tasks.

{self.plan_parser.get_format_instructions()}

Consider:
1. What are the key components of this question?
2. What types of information sources would be most valuable?
3. How should tasks be prioritized and sequenced?
4. What would constitute a successful research outcome?

Provide a structured plan that will enable thorough research of this topic.
"""
            
            response = await self.generate_response(planning_prompt, include_history=False)
            
            try:
                # Parse the structured response
                plan = self.plan_parser.parse(response)
                self.current_plan = plan
                
                # Store in research context
                self.research_context.update({
                    "original_query": query,
                    "plan": plan.dict(),
                    "plan_created_at": datetime.now().isoformat()
                })
                
                logger.info(f"Created research plan with {len(plan.tasks)} tasks")
                
                return {
                    "success": True,
                    "plan": plan.dict(),
                    "tasks_count": len(plan.tasks),
                    "estimated_duration": plan.estimated_duration,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured plan, using fallback: {parse_error}")
                
                # Fallback: extract plan information manually
                plan_data = self._extract_plan_from_text(response, query)
                
                return {
                    "success": True,
                    "plan": plan_data,
                    "tasks_count": len(plan_data.get("tasks", [])),
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to create research plan: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_plan_from_text(self, response: str, query: str) -> Dict[str, Any]:
        """
        Extract plan information from free-form text response.
        
        Args:
            response: LLM response text
            query: Original query
            
        Returns:
            Dictionary with plan information
        """
        # Simple extraction logic for fallback
        lines = response.split('\n')
        tasks = []
        
        current_task = None
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for task indicators
            if any(indicator in line.lower() for indicator in ['task', 'step', 'research', 'analyze', 'investigate']):
                if current_task:
                    tasks.append(current_task)
                
                current_task = {
                    "task_id": f"task_{task_counter}",
                    "description": line,
                    "priority": task_counter,
                    "dependencies": [],
                    "expected_sources": 5,
                    "search_terms": [query]  # Use original query as fallback
                }
                task_counter += 1
        
        if current_task:
            tasks.append(current_task)
        
        return {
            "query": query,
            "objective": f"Research and analyze: {query}",
            "tasks": tasks,
            "estimated_duration": len(tasks) * 10,  # 10 minutes per task
            "success_criteria": ["Comprehensive coverage of the topic", "Multiple reliable sources", "Clear and coherent synthesis"]
        }
    
    async def execute_research(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research plan by coordinating with research agents.
        
        Args:
            query: Research question
            context: Research context including plan
            
        Returns:
            Dictionary with research results
        """
        try:
            logger.info(f"Executing research for query: {query}")
            
            # If no plan exists, create one first
            if not self.current_plan:
                plan_result = await self.create_research_plan(query, context)
                if not plan_result.get("success"):
                    return plan_result
            
            # For now, return a structured response indicating research coordination
            # In a full implementation, this would coordinate with research agents
            
            research_response = await self.generate_response(f"""
Based on the research plan, provide a comprehensive analysis of this query: {query}

Structure your response as a detailed research report covering:

1. EXECUTIVE SUMMARY
   - Key findings and main insights
   - Direct answer to the research question

2. DETAILED ANALYSIS
   - Break down the topic into its key components
   - Provide thorough analysis of each component
   - Include relevant context and background information

3. SUPPORTING EVIDENCE
   - Reference authoritative sources and evidence
   - Explain the reasoning behind conclusions
   - Address potential counterarguments or limitations

4. CONCLUSIONS AND IMPLICATIONS
   - Summarize the main findings
   - Discuss practical implications
   - Suggest areas for further research if applicable

Ensure the response is comprehensive, well-structured, and directly addresses the original question.
""")
            
            # Store results
            self.task_results["main_research"] = {
                "query": query,
                "response": research_response,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(research_response.split())
            }
            
            return {
                "success": True,
                "research_result": research_response,
                "word_count": len(research_response.split()),
                "tasks_completed": len(self.current_plan.tasks) if self.current_plan else 1,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute research: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def synthesize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize research results into a comprehensive report.
        
        Args:
            results: List of research results from various tasks
            
        Returns:
            Dictionary with synthesized report
        """
        try:
            logger.info(f"Synthesizing results from {len(results)} sources")
            
            # Prepare synthesis context
            results_text = ""
            for i, result in enumerate(results):
                results_text += f"\n--- Source {i+1} ---\n"
                results_text += str(result.get("content", result))
                results_text += "\n"
            
            synthesis_prompt = f"""
Synthesize the following research findings into a comprehensive, well-structured report:

{results_text}

Create a synthesis that:

1. INTEGRATES all key findings coherently
2. IDENTIFIES common themes and patterns
3. RESOLVES any contradictions or conflicting information
4. PROVIDES a balanced, objective analysis
5. STRUCTURES information logically and clearly

Format as a professional research report with:
- Executive Summary
- Main Findings (with evidence)
- Analysis and Discussion
- Conclusions and Recommendations

Ensure the synthesis is authoritative, well-reasoned, and adds value beyond simply combining the sources.
"""
            
            synthesized_response = await self.generate_response(synthesis_prompt, include_history=False)
            
            return {
                "success": True,
                "synthesized_report": synthesized_response,
                "sources_count": len(results),
                "word_count": len(synthesized_response.split()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize results: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_research_status(self) -> Dict[str, Any]:
        """
        Get current research status and progress.
        
        Returns:
            Dictionary with research status information
        """
        status = {
            "has_plan": self.current_plan is not None,
            "completed_tasks": len(self.task_results),
            "research_context": self.research_context,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.current_plan:
            status.update({
                "total_tasks": len(self.current_plan.tasks),
                "plan_objective": self.current_plan.objective,
                "estimated_duration": self.current_plan.estimated_duration
            })
        
        return status
    
    def reset_research_state(self):
        """Reset the research state for a new query."""
        self.current_plan = None
        self.task_results.clear()
        self.research_context.clear()
        self.clear_history()
        logger.info("Reset research state")