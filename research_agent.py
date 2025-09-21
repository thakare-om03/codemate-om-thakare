"""
Multi-Step Reasoning Agent Framework using LangGraph
Implements deep research capabilities with query decomposition, information gathering, and synthesis
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode

# Tools and utilities
from langchain.tools import BaseTool
from langchain_core.tools import tool

from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem, HybridRetriever


class ResearchPhase(Enum):
    """Research phases for the agent workflow"""
    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    REPORTING = "reporting"


@dataclass
class ResearchQuery:
    """Structured research query"""
    original_query: str
    subqueries: List[str]
    keywords: List[str]
    research_goals: List[str]
    estimated_complexity: str  # "simple", "medium", "complex"
    required_sources: int


@dataclass
class ResearchFinding:
    """Individual research finding"""
    content: str
    source: str
    relevance_score: float
    confidence_score: float
    supporting_evidence: List[str]
    related_queries: List[str]


@dataclass
class ResearchResult:
    """Complete research result"""
    query: ResearchQuery
    findings: List[ResearchFinding]
    synthesis: str
    conclusions: List[str]
    confidence_score: float
    sources_used: List[str]
    reasoning_steps: List[str]


# Pydantic models for structured outputs
class QueryDecomposition(BaseModel):
    """Structured query decomposition"""
    subqueries: List[str] = Field(description="List of specific sub-questions to research")
    keywords: List[str] = Field(description="Key terms and concepts to search for")
    research_goals: List[str] = Field(description="Specific goals and objectives for this research")
    complexity: str = Field(description="Estimated complexity: simple, medium, or complex")
    required_sources: int = Field(description="Minimum number of sources needed")


class ResearchPlan(BaseModel):
    """Research execution plan"""
    phases: List[str] = Field(description="List of research phases to execute")
    priority_queries: List[str] = Field(description="Prioritized list of queries to investigate")
    search_strategies: List[str] = Field(description="Search strategies to employ")
    validation_criteria: List[str] = Field(description="Criteria for validating findings")


class SynthesisResult(BaseModel):
    """Synthesis of research findings"""
    main_findings: List[str] = Field(description="Key findings from the research")
    supporting_evidence: List[str] = Field(description="Evidence supporting the findings")
    conclusions: List[str] = Field(description="Final conclusions drawn from the research")
    confidence_score: float = Field(description="Overall confidence in the findings (0-1)")
    gaps_identified: List[str] = Field(description="Gaps or limitations in the research")


# LangGraph State
class ResearchState(TypedDict):
    """State for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    decomposed_query: Optional[QueryDecomposition]
    research_plan: Optional[ResearchPlan]
    current_phase: str
    findings: List[ResearchFinding]
    search_results: List[Dict[str, Any]]
    synthesis: Optional[SynthesisResult]
    final_result: Optional[ResearchResult]
    iteration_count: int
    max_iterations: int


class ResearchTools:
    """Collection of tools for the research agent"""
    
    def __init__(self, embedding_system: AdvancedEmbeddingSystem, config: ResearchConfig):
        self.embedding_system = embedding_system
        self.config = config
        self.retriever = HybridRetriever(embedding_system, config)
        
        # Create tools
        self.tools = [
            self._create_search_tool(),
            self._create_detailed_search_tool(),
            self._create_validation_tool(),
        ]
    
    def _create_search_tool(self):
        """Create basic search tool"""
        @tool
        def search_documents(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
            """Search for relevant documents in the knowledge base"""
            try:
                documents = self.retriever._get_relevant_documents(query)
                results = []
                
                for i, doc in enumerate(documents[:max_results]):
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_rank": i + 1
                    })
                
                return results
            except Exception as e:
                return [{"error": f"Search failed: {str(e)}"}]
        
        return search_documents
    
    def _create_detailed_search_tool(self):
        """Create detailed search tool with filtering"""
        @tool
        def detailed_search(query: str, keywords: List[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
            """Perform detailed search with keyword filtering and enhanced metadata"""
            try:
                # Enhance query with keywords
                enhanced_query = query
                if keywords:
                    enhanced_query += " " + " ".join(keywords)
                
                documents = self.retriever._get_relevant_documents(enhanced_query)
                results = []
                
                for i, doc in enumerate(documents[:max_results]):
                    # Calculate relevance score based on keyword matches
                    relevance_score = self._calculate_relevance(doc.page_content, keywords or [])
                    
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_rank": i + 1,
                        "relevance_score": relevance_score,
                        "keywords_found": self._find_keywords_in_text(doc.page_content, keywords or [])
                    })
                
                # Sort by relevance score
                results.sort(key=lambda x: x["relevance_score"], reverse=True)
                return results
                
            except Exception as e:
                return [{"error": f"Detailed search failed: {str(e)}"}]
        
        return detailed_search
    
    def _create_validation_tool(self):
        """Create tool for validating research findings"""
        @tool
        def validate_finding(claim: str, evidence_query: str) -> Dict[str, Any]:
            """Validate a research finding by searching for supporting evidence"""
            try:
                # Search for evidence
                evidence_docs = self.retriever._get_relevant_documents(evidence_query)
                
                supporting_evidence = []
                contradictory_evidence = []
                
                for doc in evidence_docs[:5]:
                    # Simple validation logic (could be enhanced with NLP)
                    content_lower = doc.page_content.lower()
                    claim_lower = claim.lower()
                    
                    # Check for supporting keywords
                    if any(word in content_lower for word in claim_lower.split()):
                        supporting_evidence.append({
                            "content": doc.page_content[:300] + "...",
                            "source": doc.metadata.get("source", "unknown"),
                            "confidence": 0.7  # Placeholder
                        })
                
                validation_score = len(supporting_evidence) / max(len(evidence_docs), 1)
                
                return {
                    "claim": claim,
                    "validation_score": validation_score,
                    "supporting_evidence": supporting_evidence,
                    "contradictory_evidence": contradictory_evidence,
                    "recommendation": "validated" if validation_score > 0.5 else "needs_more_evidence"
                }
                
            except Exception as e:
                return {"error": f"Validation failed: {str(e)}"}
        
        return validate_finding
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword presence"""
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return found_keywords / len(keywords)
    
    def _find_keywords_in_text(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords are present in the text"""
        text_lower = text.lower()
        return [keyword for keyword in keywords if keyword.lower() in text_lower]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return self.tools


class DeepResearchAgent:
    """Main deep research agent using LangGraph"""
    
    def __init__(self, embedding_system: AdvancedEmbeddingSystem, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.embedding_system = embedding_system
        
        # Initialize models
        model_config = self.config.get_ollama_config()
        self.llm = ChatOllama(**model_config["llm"])
        self.reasoning_llm = ChatOllama(**model_config["reasoning"])
        
        # Initialize tools
        self.research_tools = ResearchTools(embedding_system, config)
        
        # Initialize parsers
        self.decomposition_parser = PydanticOutputParser(pydantic_object=QueryDecomposition)
        self.plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        self.synthesis_parser = PydanticOutputParser(pydantic_object=SynthesisResult)
        
        # Build the graph
        self.graph = self._build_research_graph()
    
    def _build_research_graph(self) -> StateGraph:
        """Build the LangGraph research workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("decompose_query", self._decompose_query_node)
        workflow.add_node("create_research_plan", self._create_research_plan_node)
        workflow.add_node("gather_information", self._gather_information_node)
        workflow.add_node("analyze_findings", self._analyze_findings_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("validate_findings", self._validate_findings_node)
        workflow.add_node("generate_final_result", self._generate_final_result_node)
        
        # Define the workflow
        workflow.add_edge(START, "decompose_query")
        workflow.add_edge("decompose_query", "create_research_plan")
        workflow.add_edge("create_research_plan", "gather_information")
        workflow.add_edge("gather_information", "analyze_findings")
        workflow.add_edge("analyze_findings", "synthesize_results")
        workflow.add_edge("synthesize_results", "validate_findings")
        workflow.add_edge("validate_findings", "generate_final_result")
        workflow.add_edge("generate_final_result", END)
        
        # Add conditional logic for iterations
        workflow.add_conditional_edges(
            "validate_findings",
            self._should_continue_research,
            {
                "continue": "gather_information",
                "finish": "generate_final_result"
            }
        )
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _decompose_query_node(self, state: ResearchState) -> Dict[str, Any]:
        """Decompose the original query into structured components"""
        try:
            decomposition_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research assistant. Decompose the given research query into structured components.
                
                Break down the query into:
                1. Specific sub-questions that need to be answered
                2. Key terms and concepts to search for
                3. Research goals and objectives
                4. Estimated complexity level
                5. Minimum number of sources needed
                
                {format_instructions}"""),
                ("human", "Research Query: {query}")
            ])
            
            formatted_prompt = decomposition_prompt.format_messages(
                query=state["original_query"],
                format_instructions=self.decomposition_parser.get_format_instructions()
            )
            
            response = await self.reasoning_llm.ainvoke(formatted_prompt)
            decomposed_query = self.decomposition_parser.parse(response.content)
            
            return {
                "decomposed_query": decomposed_query,
                "current_phase": ResearchPhase.DECOMPOSITION.value,
                "messages": [AIMessage(content=f"Query decomposed into {len(decomposed_query.subqueries)} sub-questions")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in query decomposition: {str(e)}")]
            }
    
    async def _create_research_plan_node(self, state: ResearchState) -> Dict[str, Any]:
        """Create a detailed research plan"""
        try:
            planning_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research planning expert. Create a comprehensive research plan based on the decomposed query.
                
                Consider:
                1. Research phases to execute
                2. Priority order of sub-questions
                3. Search strategies to employ
                4. Validation criteria for findings
                
                {format_instructions}"""),
                ("human", """
                Original Query: {original_query}
                Decomposed Query: {decomposed_query}
                """)
            ])
            
            formatted_prompt = planning_prompt.format_messages(
                original_query=state["original_query"],
                decomposed_query=json.dumps(state["decomposed_query"].__dict__ if state["decomposed_query"] else {}),
                format_instructions=self.plan_parser.get_format_instructions()
            )
            
            response = await self.reasoning_llm.ainvoke(formatted_prompt)
            research_plan = self.plan_parser.parse(response.content)
            
            return {
                "research_plan": research_plan,
                "current_phase": ResearchPhase.PLANNING.value,
                "messages": [AIMessage(content=f"Research plan created with {len(research_plan.phases)} phases")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in research planning: {str(e)}")]
            }
    
    async def _gather_information_node(self, state: ResearchState) -> Dict[str, Any]:
        """Gather information using the research tools"""
        try:
            decomposed_query = state["decomposed_query"]
            if not decomposed_query:
                return {"messages": [AIMessage(content="No decomposed query available")]}
            
            all_results = []
            
            # Search for each subquery
            for subquery in decomposed_query.subqueries:
                # Use detailed search tool
                search_tool = self.research_tools.tools[1]  # detailed_search
                results = search_tool.invoke({
                    "query": subquery,
                    "keywords": decomposed_query.keywords,
                    "max_results": 5
                })
                
                all_results.extend(results)
            
            # Also search for the original query
            main_search = self.research_tools.tools[0].invoke({
                "query": state["original_query"],
                "max_results": 8
            })
            all_results.extend(main_search)
            
            return {
                "search_results": all_results,
                "current_phase": ResearchPhase.INFORMATION_GATHERING.value,
                "messages": [AIMessage(content=f"Gathered {len(all_results)} search results")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in information gathering: {str(e)}")]
            }
    
    async def _analyze_findings_node(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze the gathered information"""
        try:
            search_results = state.get("search_results", [])
            if not search_results:
                return {"messages": [AIMessage(content="No search results to analyze")]}
            
            findings = []
            
            for result in search_results:
                if "error" in result:
                    continue
                
                # Create research finding
                finding = ResearchFinding(
                    content=result.get("content", ""),
                    source=result.get("metadata", {}).get("source", "unknown"),
                    relevance_score=result.get("relevance_score", 0.5),
                    confidence_score=0.8,  # Placeholder
                    supporting_evidence=[],
                    related_queries=[]
                )
                findings.append(finding)
            
            # Sort by relevance
            findings.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return {
                "findings": findings[:self.config.MAX_DOCUMENTS_PER_QUERY],
                "current_phase": ResearchPhase.ANALYSIS.value,
                "messages": [AIMessage(content=f"Analyzed {len(findings)} findings")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in analysis: {str(e)}")]
            }
    
    async def _synthesize_results_node(self, state: ResearchState) -> Dict[str, Any]:
        """Synthesize the research findings"""
        try:
            findings = state.get("findings", [])
            if not findings:
                return {"messages": [AIMessage(content="No findings to synthesize")]}
            
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research synthesizer. Analyze the research findings and create a comprehensive synthesis.
                
                Provide:
                1. Main findings and insights
                2. Supporting evidence for each finding
                3. Clear conclusions based on the evidence
                4. Overall confidence score (0-1)
                5. Any gaps or limitations identified
                
                {format_instructions}"""),
                ("human", """
                Original Query: {original_query}
                Research Findings: {findings}
                """)
            ])
            
            findings_text = "\n\n".join([
                f"Finding {i+1}: {finding.content[:500]}... (Source: {finding.source}, Relevance: {finding.relevance_score})"
                for i, finding in enumerate(findings[:10])
            ])
            
            formatted_prompt = synthesis_prompt.format_messages(
                original_query=state["original_query"],
                findings=findings_text,
                format_instructions=self.synthesis_parser.get_format_instructions()
            )
            
            response = await self.reasoning_llm.ainvoke(formatted_prompt)
            synthesis = self.synthesis_parser.parse(response.content)
            
            return {
                "synthesis": synthesis,
                "current_phase": ResearchPhase.SYNTHESIS.value,
                "messages": [AIMessage(content=f"Synthesized findings with confidence score: {synthesis.confidence_score}")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in synthesis: {str(e)}")]
            }
    
    async def _validate_findings_node(self, state: ResearchState) -> Dict[str, Any]:
        """Validate the synthesized findings"""
        try:
            synthesis = state.get("synthesis")
            if not synthesis:
                return {"messages": [AIMessage(content="No synthesis to validate")]}
            
            validation_results = []
            
            # Validate main findings
            validation_tool = self.research_tools.tools[2]  # validate_finding
            
            for finding in synthesis.main_findings[:3]:  # Validate top 3 findings
                validation = validation_tool.invoke({
                    "claim": finding,
                    "evidence_query": f"evidence for: {finding}"
                })
                validation_results.append(validation)
            
            # Calculate overall validation score
            avg_validation_score = sum(
                result.get("validation_score", 0) for result in validation_results
            ) / max(len(validation_results), 1)
            
            return {
                "current_phase": ResearchPhase.VALIDATION.value,
                "messages": [AIMessage(content=f"Validation completed. Average score: {avg_validation_score:.2f}")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in validation: {str(e)}")]
            }
    
    async def _generate_final_result_node(self, state: ResearchState) -> Dict[str, Any]:
        """Generate the final research result"""
        try:
            decomposed_query = state.get("decomposed_query")
            findings = state.get("findings", [])
            synthesis = state.get("synthesis")
            
            if not all([decomposed_query, synthesis]):
                return {"messages": [AIMessage(content="Incomplete data for final result generation")]}
            
            # Create research query object
            research_query = ResearchQuery(
                original_query=state["original_query"],
                subqueries=decomposed_query.subqueries,
                keywords=decomposed_query.keywords,
                research_goals=decomposed_query.research_goals,
                estimated_complexity=decomposed_query.complexity,
                required_sources=decomposed_query.required_sources
            )
            
            # Create final result
            final_result = ResearchResult(
                query=research_query,
                findings=findings,
                synthesis=synthesis.main_findings[0] if synthesis.main_findings else "No synthesis available",
                conclusions=synthesis.conclusions,
                confidence_score=synthesis.confidence_score,
                sources_used=list(set(finding.source for finding in findings)),
                reasoning_steps=[msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
            )
            
            return {
                "final_result": final_result,
                "current_phase": ResearchPhase.REPORTING.value,
                "messages": [AIMessage(content="Final research result generated successfully")]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error generating final result: {str(e)}")]
            }
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Determine if research should continue or finish"""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", self.config.MAX_RESEARCH_ITERATIONS)
        
        synthesis = state.get("synthesis")
        
        # Continue if confidence is low and we haven't exceeded max iterations
        if (synthesis and 
            synthesis.confidence_score < self.config.MIN_CONFIDENCE_SCORE and 
            iteration_count < max_iterations):
            return "continue"
        
        return "finish"
    
    async def research(self, query: str, config_override: Dict[str, Any] = None) -> ResearchResult:
        """Execute the complete research workflow"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "current_phase": ResearchPhase.PLANNING.value,
                "findings": [],
                "search_results": [],
                "iteration_count": 0,
                "max_iterations": config_override.get("max_iterations", self.config.MAX_RESEARCH_ITERATIONS) if config_override else self.config.MAX_RESEARCH_ITERATIONS
            }
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return final_state.get("final_result")
            
        except Exception as e:
            print(f"Error in research execution: {e}")
            # Return a basic error result
            return ResearchResult(
                query=ResearchQuery(
                    original_query=query,
                    subqueries=[],
                    keywords=[],
                    research_goals=[],
                    estimated_complexity="unknown",
                    required_sources=0
                ),
                findings=[],
                synthesis=f"Research failed: {str(e)}",
                conclusions=[],
                confidence_score=0.0,
                sources_used=[],
                reasoning_steps=[]
            )


# Usage example
if __name__ == "__main__":
    async def main():
        from embedding_system import AdvancedEmbeddingSystem
        
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add some documents first
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Document loading result: {result}")
        
        if result.get("status") == "success":
            # Create and test the research agent
            agent = DeepResearchAgent(embedding_system, config)
            
            # Perform research
            research_result = await agent.research(
                "What are the key factors that make a good pizza restaurant according to customer reviews?"
            )
            
            print(f"\nResearch Result:")
            print(f"Query: {research_result.query.original_query}")
            print(f"Synthesis: {research_result.synthesis}")
            print(f"Conclusions: {research_result.conclusions}")
            print(f"Confidence: {research_result.confidence_score}")
            print(f"Sources: {len(research_result.sources_used)}")
    
    # Run the async function
    asyncio.run(main())