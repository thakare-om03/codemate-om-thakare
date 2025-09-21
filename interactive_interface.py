"""
Interactive Research Interface
Provides interactive query refinement and exploration capabilities
"""

import asyncio
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time

# Local imports
from config import ResearchConfig
from embedding_system import AdvancedEmbeddingSystem
from research_workflow import ResearchWorkflowOrchestrator
from report_generator import ReportGenerator, ReportMetadata
from research_agent import ResearchResult


class InteractiveResearchInterface:
    """Interactive interface for research exploration and refinement"""
    
    def __init__(self):
        self.config = ResearchConfig()
        self.config.create_directories()
        
        # Initialize components
        if 'embedding_system' not in st.session_state:
            st.session_state.embedding_system = AdvancedEmbeddingSystem(self.config)
        
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = ResearchWorkflowOrchestrator(
                st.session_state.embedding_system, 
                self.config
            )
        
        if 'report_generator' not in st.session_state:
            st.session_state.report_generator = ReportGenerator(self.config)
        
        # Initialize session state
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        
        if 'current_research' not in st.session_state:
            st.session_state.current_research = None
        
        if 'document_status' not in st.session_state:
            st.session_state.document_status = {}
    
    def render_sidebar(self):
        """Render sidebar with navigation and settings"""
        st.sidebar.title("🔬 Deep Research Agent")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Research Query", "Document Management", "Research History", "Report Generation", "System Status"]
        )
        
        # Settings
        st.sidebar.header("Settings")
        
        # Model selection
        model_options = ["llama3.2", "llama3.1", "phi3", "gemma2"]
        selected_model = st.sidebar.selectbox(
            "LLM Model",
            model_options,
            index=0
        )
        
        if selected_model != st.session_state.get('selected_model'):
            st.session_state.selected_model = selected_model
            # Update model configuration
            self.config.OLLAMA_MODELS["reasoning"]["model"] = selected_model
            self.config.OLLAMA_MODELS["summarization"]["model"] = selected_model
        
        # Research parameters
        st.sidebar.header("Research Parameters")
        
        max_findings = st.sidebar.slider(
            "Max Findings per Query",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of findings to retrieve"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence score for findings"
        )
        
        # Update session state
        st.session_state.max_findings = max_findings
        st.session_state.confidence_threshold = confidence_threshold
        
        return page
    
    def render_research_query_page(self):
        """Render main research query interface"""
        st.header("🔍 Research Query Interface")
        
        # Query input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your research question:",
                height=100,
                placeholder="What are the key factors that influence customer satisfaction in restaurants?"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Quick templates
            st.subheader("Quick Templates")
            
            if st.button("📊 Business Analysis"):
                st.session_state.template_query = "Analyze the key business factors and their impact on success metrics"
            
            if st.button("🔬 Scientific Research"):
                st.session_state.template_query = "What does current research show about the effectiveness and mechanisms"
            
            if st.button("📈 Market Analysis"):
                st.session_state.template_query = "Analyze market trends, competitive landscape, and growth opportunities"
            
            if st.button("💡 Innovation Research"):
                st.session_state.template_query = "Explore emerging trends, technologies, and innovative approaches"
        
        # Use template if selected
        if 'template_query' in st.session_state:
            query = st.session_state.template_query
            del st.session_state.template_query
        
        # Research options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_depth = st.selectbox(
                "Research Depth",
                ["Quick", "Standard", "Deep", "Comprehensive"],
                index=1
            )
        
        with col2:
            focus_areas = st.multiselect(
                "Focus Areas",
                ["Key Concepts", "Trends", "Evidence", "Comparisons", "Implications", "Future Directions"],
                default=["Key Concepts", "Evidence"]
            )
        
        with col3:
            output_style = st.selectbox(
                "Output Style",
                ["Academic", "Business", "Executive", "Detailed"],
                index=1
            )
        
        # Start research button
        if st.button("🚀 Start Research", type="primary"):
            if query.strip():
                with st.spinner("Conducting research... This may take a few minutes."):
                    asyncio.run(self._conduct_research(query, research_depth, focus_areas, output_style))
            else:
                st.error("Please enter a research question.")
        
        # Display current research results
        if st.session_state.current_research:
            self._display_research_results()
    
    async def _conduct_research(self, query: str, depth: str, focus_areas: List[str], style: str):
        """Conduct research and update session state"""
        try:
            # Configure research parameters based on depth
            depth_config = {
                "Quick": {"max_iterations": 2, "min_findings": 5},
                "Standard": {"max_iterations": 3, "min_findings": 10},
                "Deep": {"max_iterations": 4, "min_findings": 15},
                "Comprehensive": {"max_iterations": 5, "min_findings": 20}
            }
            
            config = depth_config.get(depth, depth_config["Standard"])
            
            # Add focus areas to query context
            enhanced_query = f"{query}\n\nFocus on: {', '.join(focus_areas)}"
            
            # Start research
            start_time = time.time()
            
            result = await st.session_state.orchestrator.orchestrate_research(
                enhanced_query,
                max_iterations=config["max_iterations"],
                min_findings_threshold=config["min_findings"]
            )
            
            end_time = time.time()
            
            if result['status'] == 'success':
                research_result = result['result']
                
                # Store in session state
                st.session_state.current_research = {
                    'result': research_result,
                    'query': query,
                    'depth': depth,
                    'style': style,
                    'focus_areas': focus_areas,
                    'duration': end_time - start_time,
                    'timestamp': datetime.now()
                }
                
                # Add to history
                st.session_state.research_history.append(st.session_state.current_research)
                
                st.success(f"Research completed in {end_time - start_time:.1f} seconds!")
                st.rerun()
                
            else:
                st.error(f"Research failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error conducting research: {e}")
    
    def _display_research_results(self):
        """Display current research results with interactive elements"""
        research = st.session_state.current_research
        result = research['result']
        
        st.divider()
        st.subheader("📋 Research Results")
        
        # Research summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Findings", len(result.findings))
        
        with col2:
            st.metric("Confidence Score", f"{result.confidence_score:.2f}")
        
        with col3:
            st.metric("Sources Used", len(result.sources_used))
        
        with col4:
            st.metric("Duration", f"{research['duration']:.1f}s")
        
        # Research synthesis
        st.subheader("🎯 Key Insights")
        st.write(result.synthesis)
        
        # Interactive findings exploration
        st.subheader("🔍 Explore Findings")
        
        # Create findings dataframe
        findings_data = []
        for i, finding in enumerate(result.findings):
            findings_data.append({
                'ID': i + 1,
                'Content': finding.content[:200] + "..." if len(finding.content) > 200 else finding.content,
                'Source': finding.source,
                'Relevance': finding.relevance_score,
                'Full_Content': finding.content
            })
        
        findings_df = pd.DataFrame(findings_data)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_relevance = st.slider(
                "Minimum Relevance Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        with col2:
            source_filter = st.multiselect(
                "Filter by Source",
                options=findings_df['Source'].unique(),
                default=findings_df['Source'].unique()
            )
        
        # Filter findings
        filtered_df = findings_df[
            (findings_df['Relevance'] >= min_relevance) &
            (findings_df['Source'].isin(source_filter))
        ]
        
        # Display findings
        st.dataframe(
            filtered_df[['ID', 'Content', 'Source', 'Relevance']],
            use_container_width=True,
            height=300
        )
        
        # Selected finding details
        if not filtered_df.empty:
            selected_id = st.selectbox(
                "View Full Content of Finding:",
                options=filtered_df['ID'].tolist(),
                format_func=lambda x: f"Finding {x} (Relevance: {filtered_df[filtered_df['ID']==x]['Relevance'].iloc[0]:.2f})"
            )
            
            if selected_id:
                selected_finding = filtered_df[filtered_df['ID'] == selected_id]
                st.text_area(
                    "Full Content:",
                    value=selected_finding['Full_Content'].iloc[0],
                    height=200,
                    disabled=True
                )
        
        # Visualization
        self._create_research_visualizations(result)
        
        # Follow-up questions
        self._render_followup_interface(result)
        
        # Report generation
        self._render_report_generation_interface()
    
    def _create_research_visualizations(self, result: ResearchResult):
        """Create interactive visualizations of research results"""
        st.subheader("📊 Research Analytics")
        
        # Relevance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            relevance_scores = [f.relevance_score for f in result.findings]
            
            fig = px.histogram(
                x=relevance_scores,
                nbins=20,
                title="Distribution of Relevance Scores",
                labels={'x': 'Relevance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sources breakdown
            source_counts = {}
            for finding in result.findings:
                source_counts[finding.source] = source_counts.get(finding.source, 0) + 1
            
            fig = px.pie(
                values=list(source_counts.values()),
                names=list(source_counts.keys()),
                title="Findings by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence timeline (if available)
        if hasattr(result, 'research_steps') and result.research_steps:
            steps_data = []
            for i, step in enumerate(result.research_steps):
                if hasattr(step, 'confidence'):
                    steps_data.append({
                        'Step': i + 1,
                        'Confidence': step.confidence,
                        'Description': getattr(step, 'description', f'Step {i+1}')
                    })
            
            if steps_data:
                steps_df = pd.DataFrame(steps_data)
                
                fig = px.line(
                    steps_df,
                    x='Step',
                    y='Confidence',
                    title="Research Confidence Over Steps",
                    hover_data=['Description']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_followup_interface(self, result: ResearchResult):
        """Render interface for follow-up questions and refinement"""
        st.subheader("🤔 Follow-up & Refinement")
        
        # Suggested follow-up questions
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Suggested Follow-up Questions:**")
            
            suggested_questions = [
                f"Can you provide more details about {result.query.sub_queries[0] if result.query.sub_queries else 'the main findings'}?",
                "What are the implications of these findings?",
                "Are there any contradictory viewpoints?",
                "What additional evidence supports these conclusions?",
                "How recent is this information?"
            ]
            
            for i, question in enumerate(suggested_questions):
                if st.button(question, key=f"suggested_{i}"):
                    st.session_state.followup_query = question
        
        with col2:
            st.write("**Custom Follow-up:**")
            
            followup_query = st.text_area(
                "Ask a follow-up question:",
                value=st.session_state.get('followup_query', ''),
                height=100,
                placeholder="Based on these findings, I'd like to know more about..."
            )
            
            if st.button("🔍 Research Follow-up"):
                if followup_query.strip():
                    # Combine original context with follow-up
                    enhanced_query = f"Original research: {result.query.original_query}\n\nPrevious findings summary: {result.synthesis[:500]}...\n\nFollow-up question: {followup_query}"
                    
                    with st.spinner("Researching follow-up question..."):
                        asyncio.run(self._conduct_research(
                            enhanced_query,
                            "Standard",
                            ["Evidence", "Details"],
                            "Detailed"
                        ))
    
    def _render_report_generation_interface(self):
        """Render report generation interface"""
        st.subheader("📄 Generate Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["Markdown", "HTML", "PDF", "All Formats"]
            )
        
        with col2:
            report_template = st.selectbox(
                "Report Template",
                ["Academic", "Business", "Executive", "Detailed"]
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("📄 Generate Report"):
                with st.spinner("Generating report..."):
                    asyncio.run(self._generate_report(report_format.lower(), report_template.lower()))
    
    async def _generate_report(self, format_type: str, template_type: str):
        """Generate research report"""
        try:
            result = st.session_state.current_research['result']
            
            if format_type == "all formats":
                metadata_list = await st.session_state.report_generator.generate_multiple_formats(
                    result,
                    template_type=template_type
                )
                
                st.success(f"Generated {len(metadata_list)} report formats!")
                
                for metadata in metadata_list:
                    st.download_button(
                        label=f"Download {metadata.format_type.upper()} Report",
                        data=open(metadata.file_path, 'rb').read() if metadata.format_type == 'pdf' else open(metadata.file_path, 'r', encoding='utf-8').read(),
                        file_name=Path(metadata.file_path).name,
                        mime='application/pdf' if metadata.format_type == 'pdf' else 'text/plain'
                    )
            else:
                metadata = await st.session_state.report_generator.generate_report(
                    result,
                    format_type=format_type,
                    template_type=template_type
                )
                
                st.success("Report generated successfully!")
                
                # Provide download link
                if format_type == 'pdf':
                    with open(metadata.file_path, 'rb') as f:
                        st.download_button(
                            label=f"Download {format_type.upper()} Report",
                            data=f.read(),
                            file_name=Path(metadata.file_path).name,
                            mime='application/pdf'
                        )
                else:
                    with open(metadata.file_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label=f"Download {format_type.upper()} Report",
                            data=f.read(),
                            file_name=Path(metadata.file_path).name,
                            mime='text/plain'
                        )
                
        except Exception as e:
            st.error(f"Error generating report: {e}")
    
    def render_document_management_page(self):
        """Render document management interface"""
        st.header("📚 Document Management")
        
        # Document upload
        st.subheader("📁 Add Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'json', 'md', 'html']
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save uploaded file
                        file_path = self.config.DOCUMENTS_DIR / uploaded_file.name
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        result = asyncio.run(
                            st.session_state.embedding_system.add_documents_from_file(str(file_path))
                        )
                        
                        if result['status'] == 'success':
                            st.success(f"Successfully processed {uploaded_file.name}")
                            st.session_state.document_status[uploaded_file.name] = 'processed'
                        else:
                            st.error(f"Failed to process {uploaded_file.name}: {result.get('error')}")
        
        # Document status
        st.subheader("📋 Document Status")
        
        collection_info = st.session_state.embedding_system.get_collection_stats()
        
        if collection_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", collection_info.get('count', 0))
            
            with col2:
                st.metric("Collections", len(collection_info.get('metadata', {})))
            
            with col3:
                if st.button("🗑️ Clear All Documents"):
                    if st.checkbox("Confirm deletion"):
                        asyncio.run(st.session_state.embedding_system.clear_collection())
                        st.success("All documents cleared!")
                        st.rerun()
    
    def render_research_history_page(self):
        """Render research history interface"""
        st.header("📜 Research History")
        
        if not st.session_state.research_history:
            st.info("No research history available. Start by conducting a research query.")
            return
        
        # Display research sessions
        for i, research in enumerate(reversed(st.session_state.research_history)):
            with st.expander(f"Research {len(st.session_state.research_history) - i}: {research['query'][:100]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Timestamp:** {research['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Depth:** {research['depth']}")
                    st.write(f"**Duration:** {research['duration']:.1f}s")
                
                with col2:
                    st.write(f"**Findings:** {len(research['result'].findings)}")
                    st.write(f"**Confidence:** {research['result'].confidence_score:.2f}")
                    st.write(f"**Sources:** {len(research['result'].sources_used)}")
                
                st.write("**Focus Areas:**", ", ".join(research['focus_areas']))
                
                # Synthesis preview
                st.write("**Synthesis Preview:**")
                st.write(research['result'].synthesis[:300] + "..." if len(research['result'].synthesis) > 300 else research['result'].synthesis)
                
                # Load session button
                if st.button(f"Load Research Session {len(st.session_state.research_history) - i}", key=f"load_{i}"):
                    st.session_state.current_research = research
                    st.success("Research session loaded!")
                    st.rerun()
    
    def render_report_generation_page(self):
        """Render report generation management"""
        st.header("📊 Report Generation")
        
        # List existing reports
        reports = st.session_state.report_generator.list_reports()
        
        if reports:
            st.subheader("📋 Generated Reports")
            
            for report in reports:
                with st.expander(f"{report['title']} ({report['format_type'].upper()})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Date:** {report['created_date']}")
                        st.write(f"**Format:** {report['format_type'].upper()}")
                        st.write(f"**Confidence:** {report['confidence_score']:.2f}")
                    
                    with col2:
                        st.write(f"**Sources:** {report['total_sources']}")
                        st.write(f"**Query:** {report['research_query'][:100]}...")
                    
                    # Download button
                    if Path(report['file_path']).exists():
                        if report['format_type'] == 'pdf':
                            with open(report['file_path'], 'rb') as f:
                                st.download_button(
                                    label="Download Report",
                                    data=f.read(),
                                    file_name=Path(report['file_path']).name,
                                    mime='application/pdf'
                                )
                        else:
                            with open(report['file_path'], 'r', encoding='utf-8') as f:
                                st.download_button(
                                    label="Download Report",
                                    data=f.read(),
                                    file_name=Path(report['file_path']).name,
                                    mime='text/plain'
                                )
        else:
            st.info("No reports generated yet. Complete a research query and generate a report.")
    
    def render_system_status_page(self):
        """Render system status and diagnostics"""
        st.header("⚙️ System Status")
        
        # System health checks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Component Status")
            
            # Check Ollama connection
            try:
                # Test LLM connection
                test_result = asyncio.run(self._test_ollama_connection())
                st.success("✅ Ollama LLM: Connected") if test_result else st.error("❌ Ollama LLM: Failed")
            except:
                st.error("❌ Ollama LLM: Failed")
            
            # Check ChromaDB
            try:
                collection_info = asyncio.run(st.session_state.embedding_system.get_collection_info())
                st.success("✅ ChromaDB: Connected") if collection_info else st.error("❌ ChromaDB: Failed")
            except:
                st.error("❌ ChromaDB: Failed")
            
            # Check directories
            st.success("✅ Directories: Configured") if self.config.DOCUMENTS_DIR.exists() else st.error("❌ Directories: Missing")
        
        with col2:
            st.subheader("📊 Usage Statistics")
            
            # Research statistics
            st.metric("Total Research Sessions", len(st.session_state.research_history))
            
            if st.session_state.research_history:
                avg_confidence = sum(r['result'].confidence_score for r in st.session_state.research_history) / len(st.session_state.research_history)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
                
                total_findings = sum(len(r['result'].findings) for r in st.session_state.research_history)
                st.metric("Total Findings", total_findings)
        
        # System configuration
        st.subheader("⚙️ Configuration")
        
        with st.expander("View Configuration"):
            config_dict = {
                'Ollama Models': self.config.OLLAMA_MODELS,
                'ChromaDB Settings': {
                    'collection_name': self.config.CHROMA_COLLECTION_NAME,
                    'persistence_directory': str(self.config.CHROMA_PERSIST_DIRECTORY)
                },
                'Directories': {
                    'documents': str(self.config.DOCUMENTS_DIR),
                    'reports': str(self.config.REPORTS_DIR)
                }
            }
            st.json(config_dict)
    
    async def _test_ollama_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
            response = await llm.ainvoke("Test connection")
            return True
        except:
            return False
    
    def run(self):
        """Run the interactive interface"""
        st.set_page_config(
            page_title="Deep Research Agent",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize interface
        page = self.render_sidebar()
        
        # Render selected page
        if page == "Research Query":
            self.render_research_query_page()
        elif page == "Document Management":
            self.render_document_management_page()
        elif page == "Research History":
            self.render_research_history_page()
        elif page == "Report Generation":
            self.render_report_generation_page()
        elif page == "System Status":
            self.render_system_status_page()


# CLI Interface for non-Streamlit usage
class CLIInterface:
    """Command-line interface for the research agent"""
    
    def __init__(self):
        self.config = ResearchConfig()
        self.config.create_directories()
        
        self.embedding_system = AdvancedEmbeddingSystem(self.config)
        self.orchestrator = ResearchWorkflowOrchestrator(self.embedding_system, self.config)
        self.report_generator = ReportGenerator(self.config)
    
    async def interactive_session(self):
        """Run interactive CLI session"""
        print("🔬 Deep Research Agent - CLI Interface")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Add documents")
            print("2. Conduct research")
            print("3. Generate report")
            print("4. View research history")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                await self._add_documents()
            elif choice == "2":
                await self._conduct_research()
            elif choice == "3":
                await self._generate_report()
            elif choice == "4":
                await self._view_history()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
    
    async def _add_documents(self):
        """Add documents via CLI"""
        file_path = input("Enter file path: ").strip()
        
        if Path(file_path).exists():
            print(f"Processing {file_path}...")
            result = await self.embedding_system.add_documents_from_file(file_path)
            
            if result['status'] == 'success':
                print(f"✅ Successfully processed {result['document_count']} documents")
            elif result['status'] == 'skipped':
                print(f"⏭️ Skipped: {result['reason']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print("❌ File not found")
    
    async def _conduct_research(self):
        """Conduct research via CLI"""
        query = input("Enter research question: ").strip()
        
        if query:
            print("Conducting research...")
            result = await self.orchestrator.orchestrate_research(query)
            
            if result['status'] == 'success':
                research_result = result['result']
                print(f"\n✅ Research completed!")
                print(f"Findings: {len(research_result.findings)}")
                print(f"Confidence: {research_result.confidence_score:.2f}")
                print(f"Sources: {len(research_result.sources_used)}")
                print(f"\nSynthesis:\n{research_result.synthesis}")
                
                # Store for report generation
                self.last_research = research_result
            else:
                print(f"❌ Research failed: {result['error']}")
    
    async def _generate_report(self):
        """Generate report via CLI"""
        if not hasattr(self, 'last_research'):
            print("❌ No research results available. Please conduct research first.")
            return
        
        format_type = input("Report format (markdown/html/pdf) [markdown]: ").strip() or "markdown"
        template = input("Template (academic/business/executive/detailed) [detailed]: ").strip() or "detailed"
        
        print("Generating report...")
        metadata = await self.report_generator.generate_report(
            self.last_research,
            format_type=format_type,
            template_type=template
        )
        
        print(f"✅ Report generated: {metadata.file_path}")
    
    async def _view_history(self):
        """View research history via CLI"""
        reports = self.report_generator.list_reports()
        
        if reports:
            print("\nGenerated Reports:")
            for i, report in enumerate(reports, 1):
                print(f"{i}. {report['title']} ({report['format_type']} - {report['created_date']})")
        else:
            print("No reports found.")


# Entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run CLI interface
        cli = CLIInterface()
        asyncio.run(cli.interactive_session())
    else:
        # Run Streamlit interface
        interface = InteractiveResearchInterface()
        interface.run()