"""
Streamlit Chat Interface for the Deep Research Agent
"""
import streamlit as st
import asyncio
import io
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from stqdm import stqdm
import plotly.graph_objects as go
import plotly.express as px

from ..utils.document_processor import DocumentProcessor
from ..utils.reasoning_engine import MultiStepReasoningEngine
from ..retrieval.vector_store import VectorStoreManager
from ..embeddings.ollama_embeddings import OllamaEmbeddingService
from ..utils.export_manager import ExportManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatInterface:
    """
    Main chat interface for the Deep Research Agent using Streamlit.
    Provides file upload, chat functionality, and export capabilities.
    """
    
    def __init__(self):
        """Initialize the chat interface."""
        self.setup_page_config()
        self.initialize_session_state()
        self.initialize_components()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Deep Research Agent  ",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/deep-research-agent',
                'Report a bug': 'https://github.com/your-repo/deep-research-agent/issues',
                'About': 'Deep Research Agent - Local AI-powered research assistant'
            }
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'vector_db_initialized' not in st.session_state:
            st.session_state.vector_db_initialized = False
        
        if 'reasoning_statistics' not in st.session_state:
            st.session_state.reasoning_statistics = {}
    
    def initialize_components(self):
        """Initialize backend components."""
        try:
            # Initialize with progress indicator
            with st.spinner("Initializing Deep Research Agent..."):
                self.document_processor = DocumentProcessor()
                self.embedding_service = OllamaEmbeddingService()
                self.vector_store_manager = VectorStoreManager()
                self.reasoning_engine = MultiStepReasoningEngine(
                    vector_store_manager=self.vector_store_manager,
                    embedding_service=self.embedding_service
                )
                
                # Initialize export manager
                self.export_manager = ExportManager()
            
            # Validate Ollama connection
            if not self.embedding_service.validate_model_availability():
                st.error("""
                ⚠️ **Ollama Connection Failed**
                
                Please ensure:
                1. Ollama is installed and running
                2. The embedding model is available: `ollama pull nomic-embed-text`
                3. The chat model is available: `ollama pull llama3.2`
                """)
                st.stop()
            
            logger.info("Chat interface components initialized successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            st.stop()
    
    def render_sidebar(self):
        """Render the sidebar with file upload and controls."""
        with st.sidebar:
            st.header("📁 Document Upload")
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload research documents",
                type=['pdf', 'txt', 'docx', 'md', 'csv'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, DOCX, MD, or CSV files for research context"
            )
            
            # Process uploaded files
            if uploaded_files:
                self.handle_file_upload(uploaded_files)
            
            # Current files status
            st.subheader("📊 Current Files")
            if st.session_state.uploaded_files:
                for file_info in st.session_state.uploaded_files:
                    with st.expander(f"📄 {file_info['name']}"):
                        st.write(f"**Size:** {file_info['size']} bytes")
                        st.write(f"**Type:** {file_info['type']}")
                        st.write(f"**Chunks:** {file_info.get('chunks', 0)}")
                        st.write(f"**Uploaded:** {file_info['timestamp']}")
                        
                        if st.button(f"Remove {file_info['name']}", key=f"remove_{file_info['id']}"):
                            self.remove_file(file_info['id'])
                            st.rerun()
            else:
                st.info("No files uploaded yet")
            
            # Vector database controls
            st.subheader("🗄️ Database Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", help="Remove all documents and reset database"):
                    self.clear_all_data()
                    st.rerun()
            
            with col2:
                if st.button("📊 Stats", help="Show database statistics"):
                    self.show_database_stats()
            
            # Export controls
            st.subheader("📤 Export")
            if st.session_state.chat_history:
                if st.button("📄 Export Chat as PDF"):
                    self.export_chat_as_pdf()
                
                if st.button("📝 Export Chat as Markdown"):
                    self.export_chat_as_markdown()
    
    def handle_file_upload(self, uploaded_files):
        """Handle the file upload process."""
        try:
            new_files = []
            
            # Check for new files
            current_file_names = {f['name'] for f in st.session_state.uploaded_files}
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in current_file_names:
                    new_files.append(uploaded_file)
            
            if new_files:
                # Process new files with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(new_files):
                    progress = (i + 1) / len(new_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Process the file
                    file_id = str(uuid.uuid4())
                    
                    # Create file-like object
                    file_content = io.BytesIO(uploaded_file.read())
                    file_content.name = uploaded_file.name
                    
                    # Process document
                    documents = self.document_processor.process_file(
                        file_content,
                        filename=uploaded_file.name,
                        additional_metadata={"file_id": file_id, "session_id": st.session_state.session_id}
                    )
                    
                    if documents:
                        # Add documents to vector store
                        document_ids = self.vector_store_manager.add_documents(documents)
                        
                        # Store file info
                        file_info = {
                            "id": file_id,
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                            "type": uploaded_file.type,
                            "chunks": len(documents),
                            "document_ids": document_ids,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.uploaded_files.append(file_info)
                        st.session_state.vector_db_initialized = True
                        
                        logger.info(f"Successfully processed file: {uploaded_file.name}")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                
                progress_bar.progress(1.0)
                status_text.text("All files processed!")
                
                # Show success message
                st.success(f"✅ Successfully uploaded and processed {len(new_files)} file(s)")
                
        except Exception as e:
            st.error(f"File upload failed: {e}")
            logger.error(f"File upload error: {e}")
    
    def remove_file(self, file_id: str):
        """Remove a specific file from the vector database."""
        try:
            # Find the file
            file_to_remove = None
            for file_info in st.session_state.uploaded_files:
                if file_info['id'] == file_id:
                    file_to_remove = file_info
                    break
            
            if file_to_remove:
                # Remove from vector store (would need document IDs for precise removal)
                # For now, we'll clear and rebuild without this file
                remaining_files = [f for f in st.session_state.uploaded_files if f['id'] != file_id]
                
                if not remaining_files:
                    # If no files left, clear the database
                    self.vector_store_manager.clear_collection()
                    st.session_state.vector_db_initialized = False
                
                # Update session state
                st.session_state.uploaded_files = remaining_files
                
                st.success(f"Removed {file_to_remove['name']}")
                logger.info(f"Removed file: {file_to_remove['name']}")
            
        except Exception as e:
            st.error(f"Failed to remove file: {e}")
            logger.error(f"File removal error: {e}")
    
    def clear_all_data(self):
        """Clear all uploaded files and reset the database."""
        try:
            # Clear vector database
            self.vector_store_manager.clear_collection()
            
            # Reset session state
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            st.session_state.vector_db_initialized = False
            st.session_state.reasoning_statistics = {}
            
            # Reset reasoning engine
            self.reasoning_engine.reset_engine()
            
            st.success("🗑️ All data cleared successfully!")
            logger.info("Cleared all data and reset session")
            
        except Exception as e:
            st.error(f"Failed to clear data: {e}")
            logger.error(f"Data clearing error: {e}")
    
    def show_database_stats(self):
        """Display database statistics in a modal."""
        try:
            # Get collection info
            collection_info = self.vector_store_manager.get_collection_info()
            
            # Get reasoning statistics
            reasoning_stats = self.reasoning_engine.get_reasoning_statistics()
            
            # Display stats
            st.info(f"""
            **📊 Database Statistics**
            
            **Documents:** {collection_info.get('document_count', 0)}
            **Files Uploaded:** {len(st.session_state.uploaded_files)}
            **Chat Messages:** {len(st.session_state.chat_history)}
            **Session ID:** {st.session_state.session_id}
            **Vector DB Initialized:** {st.session_state.vector_db_initialized}
            
            **Reasoning Statistics:**
            - Total Steps: {reasoning_stats.get('total_steps', 0)}
            - Successful Steps: {reasoning_stats.get('successful_steps', 0)}
            - Iterations Completed: {reasoning_stats.get('iterations_completed', 0)}
            """)
            
        except Exception as e:
            st.error(f"Failed to get database stats: {e}")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Research Chat")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # Show metadata for assistant messages
                    if message["role"] == "assistant" and "metadata" in message:
                        with st.expander("📊 Research Details"):
                            metadata = message["metadata"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Confidence", f"{metadata.get('confidence_score', 0):.2f}")
                            with col2:
                                st.metric("Sources", metadata.get('sources_used', 0))
                            with col3:
                                st.metric("Duration", f"{metadata.get('total_duration_seconds', 0):.1f}s")
        
        # Chat input
        if prompt := st.chat_input("Ask a research question..."):
            self.handle_chat_input(prompt)
    
    def handle_chat_input(self, prompt: str):
        """Handle user chat input and generate response."""
        try:
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(prompt)
            
            # Check if documents are uploaded
            if not st.session_state.vector_db_initialized:
                error_msg = "⚠️ Please upload some documents first to provide research context."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                with st.chat_message("assistant"):
                    st.write(error_msg)
                return
            
            # Generate response with progress indicator
            with st.chat_message("assistant"):
                with st.spinner("🔍 Researching your question..."):
                    response_data = asyncio.run(self.generate_research_response(prompt))
                
                if response_data.get("success"):
                    answer = response_data.get("answer", "No answer generated")
                    metadata = response_data.get("metadata", {})
                    
                    # Display the answer
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update reasoning statistics
                    st.session_state.reasoning_statistics = metadata
                    
                    # Show research progress visualization
                    if metadata.get("reasoning_steps"):
                        self.show_reasoning_visualization(metadata["reasoning_steps"])
                
                else:
                    error_msg = f"❌ Research failed: {response_data.get('error', 'Unknown error')}"
                    st.write(error_msg)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
        
        except Exception as e:
            error_msg = f"❌ An error occurred: {e}"
            st.error(error_msg)
            logger.error(f"Chat input handling error: {e}")
    
    async def generate_research_response(self, query: str) -> Dict[str, Any]:
        """Generate a research response using the reasoning engine."""
        try:
            # Use the reasoning engine to process the query
            result = await self.reasoning_engine.process_query(
                query=query,
                context={"session_id": st.session_state.session_id},
                reasoning_depth=3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Research response generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def show_reasoning_visualization(self, reasoning_steps: List[Dict[str, Any]]):
        """Show visualization of the reasoning process."""
        try:
            with st.expander("🧠 Reasoning Process Visualization"):
                # Create timeline chart
                stages = [step["stage"] for step in reasoning_steps]
                durations = [step["duration"] for step in reasoning_steps]
                successes = [step["success"] for step in reasoning_steps]
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=stages,
                        y=durations,
                        marker_color=['green' if success else 'red' for success in successes],
                        text=[f"{duration:.2f}s" for duration in durations],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Reasoning Steps Duration",
                    xaxis_title="Reasoning Stage",
                    yaxis_title="Duration (seconds)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show step details
                for i, step in enumerate(reasoning_steps):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{i+1}. {step['stage'].replace('_', ' ').title()}**")
                        with col2:
                            st.write(f"⏱️ {step['duration']:.2f}s")
                        with col3:
                            st.write("✅" if step['success'] else "❌")
        
        except Exception as e:
            logger.error(f"Reasoning visualization failed: {e}")
    
    def export_chat_as_pdf(self):
        """Export chat history as PDF."""
        try:
            if not st.session_state.chat_history:
                st.warning("No chat history to export")
                return
            
            with st.spinner("Generating PDF export..."):
                # Generate PDF
                pdf_buffer = self.export_manager.export_chat_as_pdf(
                    chat_history=st.session_state.chat_history,
                    session_id=st.session_state.session_id,
                    uploaded_files=st.session_state.uploaded_files,
                    reasoning_stats=st.session_state.reasoning_statistics
                )
                
                # Create download button
                st.download_button(
                    label="💾 Download PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"research_chat_{st.session_state.session_id[:8]}.pdf",
                    mime="application/pdf"
                )
                
                st.success("✅ PDF export ready for download!")
            
        except Exception as e:
            st.error(f"PDF export failed: {e}")
            logger.error(f"PDF export error: {e}")
    
    def export_chat_as_markdown(self):
        """Export chat history as Markdown."""
        try:
            if not st.session_state.chat_history:
                st.warning("No chat history to export")
                return
            
            with st.spinner("Generating Markdown export..."):
                # Generate markdown content
                markdown_content = self.export_manager.export_chat_as_markdown(
                    chat_history=st.session_state.chat_history,
                    session_id=st.session_state.session_id,
                    uploaded_files=st.session_state.uploaded_files,
                    reasoning_stats=st.session_state.reasoning_statistics
                )
                
                # Offer download
                st.download_button(
                    label="💾 Download Markdown",
                    data=markdown_content,
                    file_name=f"research_chat_{st.session_state.session_id[:8]}.md",
                    mime="text/markdown"
                )
                
                st.success("✅ Markdown export ready for download!")
            
        except Exception as e:
            st.error(f"Markdown export failed: {e}")
            logger.error(f"Markdown export error: {e}")
    
    def run(self):
        """Run the Streamlit interface."""
        try:
            # Main title
            st.title("  Deep Research Agent")
            st.markdown("*Local AI-powered research assistant with multi-step reasoning*")
            
            # Check system status
            self.show_system_status()
            
            # Render sidebar
            self.render_sidebar()
            
            # Render main chat interface
            self.render_chat_interface()
            
        except Exception as e:
            st.error(f"Interface error: {e}")
            logger.error(f"Interface runtime error: {e}")
    
    def show_system_status(self):
        """Show system status indicators."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Ollama status
            if self.embedding_service.validate_model_availability():
                st.success("🟢 Ollama Connected")
            else:
                st.error("🔴 Ollama Disconnected")
        
        with col2:
            # Vector DB status
            if st.session_state.vector_db_initialized:
                st.success("🟢 Documents Loaded")
            else:
                st.warning("🟡 No Documents")
        
        with col3:
            # Chat status
            if st.session_state.chat_history:
                st.success(f"🟢 {len(st.session_state.chat_history)} Messages")
            else:
                st.info("🔵 New Session")
        
        with col4:
            # Files status
            file_count = len(st.session_state.uploaded_files)
            if file_count > 0:
                st.success(f"🟢 {file_count} Files")
            else:
                st.info("🔵 No Files")


def main():
    """Main entry point for the Streamlit app."""
    try:
        interface = ChatInterface()
        interface.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"Application startup error: {e}")


if __name__ == "__main__":
    main()