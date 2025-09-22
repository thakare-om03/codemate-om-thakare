"""
Base agent class for the multi-agent research system
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the research system.
    Provides common functionality for agent communication and execution.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            model: Ollama model to use
            base_url: Ollama server URL
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
        """
        self.name = name
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.system_prompt = system_prompt
        
        # Initialize the language model
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        # Agent state
        self.agent_id = str(uuid.uuid4())
        self.conversation_history: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "agent_type": self.__class__.__name__,
            "model": model,
            "temperature": temperature
        }
        
        logger.info(f"Initialized {self.__class__.__name__} '{name}' with model '{model}'")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation history."""
        self.conversation_history.append(SystemMessage(content=content))
    
    def add_human_message(self, content: str):
        """Add a human message to the conversation history."""
        self.conversation_history.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add an AI message to the conversation history."""
        self.conversation_history.append(AIMessage(content=content))
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.debug(f"Cleared conversation history for agent '{self.name}'")
    
    def get_conversation_context(self, max_messages: int = 10) -> List[BaseMessage]:
        """
        Get recent conversation context.
        
        Args:
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-max_messages:] if max_messages > 0 else self.conversation_history
    
    async def generate_response(
        self,
        prompt: str,
        include_history: bool = True,
        max_history: int = 5
    ) -> str:
        """
        Generate a response using the language model.
        
        Args:
            prompt: Input prompt
            include_history: Whether to include conversation history
            max_history: Maximum number of history messages to include
            
        Returns:
            Generated response text
        """
        try:
            messages = []
            
            # Add system prompt if available
            if self.system_prompt:
                messages.append(SystemMessage(content=self.system_prompt))
            
            # Add conversation history if requested
            if include_history and self.conversation_history:
                recent_history = self.get_conversation_context(max_history)
                messages.extend(recent_history)
            
            # Add the current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            # Store in conversation history
            self.add_human_message(prompt)
            self.add_ai_message(response.content)
            
            logger.debug(f"Agent '{self.name}' generated response: {len(response.content)} characters")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate response for agent '{self.name}': {e}")
            raise
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with given arguments.
        
        Args:
            template: Prompt template string
            **kwargs: Template variables
            
        Returns:
            Formatted prompt
        """
        try:
            prompt_template = PromptTemplate.from_template(template)
            return prompt_template.format(**kwargs)
        except Exception as e:
            logger.error(f"Failed to format prompt: {e}")
            return template
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.__class__.__name__,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools_count": len(self.tools),
            "conversation_length": len(self.conversation_history),
            "metadata": self.metadata
        }
    
    def set_metadata(self, key: str, value: Any):
        """Set a metadata key-value pair."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key."""
        return self.metadata.get(key, default)


class AgentCommunicationManager:
    """
    Manager for handling communication between agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[Dict[str, Any]] = []
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the communication manager."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent '{agent.name}' with communication manager")
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the communication manager."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent '{agent_name}' from communication manager")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "query"
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Sender agent name
            to_agent: Receiver agent name
            message: Message content
            message_type: Type of message
            
        Returns:
            Response from the target agent
        """
        try:
            # Get target agent
            target_agent = self.get_agent(to_agent)
            if not target_agent:
                raise ValueError(f"Agent '{to_agent}' not found")
            
            # Log the communication
            communication_log = {
                "timestamp": datetime.now().isoformat(),
                "from": from_agent,
                "to": to_agent,
                "message": message,
                "type": message_type
            }
            self.message_history.append(communication_log)
            
            logger.debug(f"Agent communication: {from_agent} -> {to_agent}")
            
            # Process the message with the target agent
            response = await target_agent.process({
                "message": message,
                "from_agent": from_agent,
                "message_type": message_type
            })
            
            # Log the response
            response_log = {
                "timestamp": datetime.now().isoformat(),
                "from": to_agent,
                "to": from_agent,
                "message": response.get("response", ""),
                "type": "response"
            }
            self.message_history.append(response_log)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send message from '{from_agent}' to '{to_agent}': {e}")
            raise
    
    def get_communication_history(
        self,
        agent_name: Optional[str] = None,
        max_messages: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get communication history.
        
        Args:
            agent_name: Filter by agent name (None for all)
            max_messages: Maximum number of messages to return
            
        Returns:
            List of communication log entries
        """
        history = self.message_history
        
        if agent_name:
            history = [
                msg for msg in history
                if msg.get("from") == agent_name or msg.get("to") == agent_name
            ]
        
        return history[-max_messages:] if max_messages > 0 else history
    
    def clear_history(self):
        """Clear all communication history."""
        self.message_history.clear()
        logger.info("Cleared communication history")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent communications."""
        stats = {
            "total_agents": len(self.agents),
            "total_messages": len(self.message_history),
            "agent_list": list(self.agents.keys())
        }
        
        # Message counts per agent
        agent_message_counts = {}
        for msg in self.message_history:
            from_agent = msg.get("from", "unknown")
            to_agent = msg.get("to", "unknown")
            
            if from_agent not in agent_message_counts:
                agent_message_counts[from_agent] = {"sent": 0, "received": 0}
            if to_agent not in agent_message_counts:
                agent_message_counts[to_agent] = {"sent": 0, "received": 0}
            
            agent_message_counts[from_agent]["sent"] += 1
            agent_message_counts[to_agent]["received"] += 1
        
        stats["agent_message_counts"] = agent_message_counts
        
        return stats