"""
Export utilities for chat sessions and research results
"""
import io
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import markdown
from fpdf import FPDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.colors import Color, black, blue, gray

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExportManager:
    """
    Manages export functionality for chat sessions and research results.
    Supports PDF and Markdown export formats.
    """
    
    def __init__(self):
        """Initialize the export manager."""
        self.setup_styles()
    
    def setup_styles(self):
        """Setup document styles for PDF export."""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ChatUser',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leftIndent=0,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ChatAssistant',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            leftIndent=20,
            textColor=Color(0.2, 0.2, 0.2),
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=6,
            leftIndent=40,
            textColor=gray,
            fontName='Helvetica-Oblique'
        ))
    
    def export_chat_as_markdown(
        self,
        chat_history: List[Dict[str, Any]],
        session_id: str,
        uploaded_files: List[Dict[str, Any]] = None,
        reasoning_stats: Dict[str, Any] = None
    ) -> str:
        """
        Export chat history as Markdown format.
        
        Args:
            chat_history: List of chat messages
            session_id: Session identifier
            uploaded_files: List of uploaded file information
            reasoning_stats: Reasoning engine statistics
        
        Returns:
            Markdown formatted string
        """
        try:
            markdown_content = []
            
            # Header
            markdown_content.append("#   Deep Research Agent - Chat Session")
            markdown_content.append("")
            
            # Metadata
            markdown_content.append("## 📊 Session Information")
            markdown_content.append("")
            markdown_content.append(f"**Session ID:** `{session_id}`")
            markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            markdown_content.append(f"**Total Messages:** {len(chat_history)}")
            
            # File information
            if uploaded_files:
                markdown_content.append("")
                markdown_content.append("### 📁 Uploaded Files")
                markdown_content.append("")
                for file_info in uploaded_files:
                    markdown_content.append(f"- **{file_info['name']}** ({file_info['type']}, {file_info['size']} bytes, {file_info.get('chunks', 0)} chunks)")
            
            # Reasoning statistics
            if reasoning_stats:
                markdown_content.append("")
                markdown_content.append("### 🧠 Reasoning Statistics")
                markdown_content.append("")
                for key, value in reasoning_stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    markdown_content.append(f"- **{formatted_key}:** {value}")
            
            markdown_content.append("")
            markdown_content.append("---")
            markdown_content.append("")
            
            # Chat messages
            markdown_content.append("## 💬 Chat History")
            markdown_content.append("")
            
            for i, message in enumerate(chat_history, 1):
                timestamp = message.get('timestamp', '')
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                
                # Message header
                if role == 'user':
                    emoji = "👤"
                    role_text = "**You**"
                else:
                    emoji = "🤖"
                    role_text = "**Research Agent**"
                
                markdown_content.append(f"### {emoji} {role_text} - Message {i}")
                
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%H:%M:%S')
                        markdown_content.append(f"*{formatted_time}*")
                    except:
                        markdown_content.append(f"*{timestamp}*")
                
                markdown_content.append("")
                
                # Message content
                # Clean content for markdown (remove HTML tags)
                import re
                from html import unescape
                
                cleaned_content = content
                try:
                    # Remove HTML tags and unescape HTML entities
                    cleaned_content = re.sub(r'<[^>]+>', '', content)
                    cleaned_content = unescape(cleaned_content)
                except Exception as clean_error:
                    logger.warning(f"Content cleaning for markdown failed: {clean_error}")
                    cleaned_content = str(content)
                
                markdown_content.append(cleaned_content)
                markdown_content.append("")
                
                # Metadata for assistant messages
                if role == 'assistant' and 'metadata' in message:
                    metadata = message['metadata']
                    
                    markdown_content.append("#### 📊 Research Details")
                    markdown_content.append("")
                    
                    # Key metrics
                    confidence = metadata.get('confidence_score', 0)
                    sources = metadata.get('sources_used', 0)
                    duration = metadata.get('total_duration_seconds', 0)
                    
                    markdown_content.append(f"- **Confidence Score:** {confidence:.2f}")
                    markdown_content.append(f"- **Sources Used:** {sources}")
                    markdown_content.append(f"- **Processing Duration:** {duration:.1f} seconds")
                    
                    # Reasoning steps
                    if 'reasoning_steps' in metadata:
                        markdown_content.append("")
                        markdown_content.append("##### 🧠 Reasoning Steps")
                        markdown_content.append("")
                        
                        for j, step in enumerate(metadata['reasoning_steps'], 1):
                            stage = step.get('stage', '').replace('_', ' ').title()
                            duration = step.get('duration', 0)
                            success = step.get('success', False)
                            status = "✅" if success else "❌"
                            
                            markdown_content.append(f"{j}. **{stage}** ({duration:.2f}s) {status}")
                    
                    markdown_content.append("")
                
                markdown_content.append("---")
                markdown_content.append("")
            
            # Footer
            markdown_content.append("*Generated by Deep Research Agent - Local AI Research Assistant*")
            
            return "\n".join(markdown_content)
            
        except Exception as e:
            logger.error(f"Markdown export failed: {e}")
            raise Exception(f"Failed to export chat as Markdown: {e}")
    
    def export_chat_as_pdf(
        self,
        chat_history: List[Dict[str, Any]],
        session_id: str,
        uploaded_files: List[Dict[str, Any]] = None,
        reasoning_stats: Dict[str, Any] = None,
        output_path: Optional[Path] = None
    ) -> io.BytesIO:
        """
        Export chat history as PDF format.
        
        Args:
            chat_history: List of chat messages
            session_id: Session identifier
            uploaded_files: List of uploaded file information
            reasoning_stats: Reasoning engine statistics
            output_path: Optional path to save PDF file
        
        Returns:
            BytesIO object containing PDF data
        """
        try:
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content
            story = []
            
            # Title
            title = Paragraph("  Deep Research Agent - Chat Session", self.styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Session information
            session_info = [
                f"<b>Session ID:</b> {session_id}",
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"<b>Total Messages:</b> {len(chat_history)}"
            ]
            
            for info in session_info:
                story.append(Paragraph(info, self.styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            # File information
            if uploaded_files:
                story.append(Paragraph("<b>📁 Uploaded Files</b>", self.styles['Heading2']))
                story.append(Spacer(1, 6))
                
                for file_info in uploaded_files:
                    file_text = f"• <b>{file_info['name']}</b> ({file_info['type']}, {file_info['size']} bytes, {file_info.get('chunks', 0)} chunks)"
                    story.append(Paragraph(file_text, self.styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            # Reasoning statistics
            if reasoning_stats:
                story.append(Paragraph("<b>🧠 Reasoning Statistics</b>", self.styles['Heading2']))
                story.append(Spacer(1, 6))
                
                for key, value in reasoning_stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    stat_text = f"• <b>{formatted_key}:</b> {value}"
                    story.append(Paragraph(stat_text, self.styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            # Chat messages
            story.append(Paragraph("<b>💬 Chat History</b>", self.styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for i, message in enumerate(chat_history, 1):
                timestamp = message.get('timestamp', '')
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                
                # Message header
                if role == 'user':
                    emoji = "👤"
                    role_text = "You"
                    style = self.styles['ChatUser']
                else:
                    emoji = "🤖"
                    role_text = "Research Agent"
                    style = self.styles['ChatAssistant']
                
                # Format timestamp
                time_text = ""
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_text = f" - {dt.strftime('%H:%M:%S')}"
                    except:
                        time_text = f" - {timestamp}"
                
                # Message header
                header_text = f"<b>{emoji} {role_text} - Message {i}{time_text}</b>"
                story.append(Paragraph(header_text, self.styles['Heading3']))
                story.append(Spacer(1, 6))
                
                # Message content
                # Clean and format content for PDF
                try:
                    cleaned_content = self._clean_content_for_pdf(content)
                    story.append(Paragraph(cleaned_content, style))
                except Exception as content_error:
                    logger.error(f"Error processing message content for PDF: {content_error}")
                    # Fallback to plain text
                    plain_content = str(content).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                    if len(plain_content) > 1000:
                        plain_content = plain_content[:1000] + "... (content simplified due to formatting error)"
                    story.append(Paragraph(plain_content, style))
                story.append(Spacer(1, 6))
                
                # Metadata for assistant messages
                if role == 'assistant' and 'metadata' in message:
                    metadata = message['metadata']
                    
                    # Key metrics
                    confidence = metadata.get('confidence_score', 0)
                    sources = metadata.get('sources_used', 0)
                    duration = metadata.get('total_duration_seconds', 0)
                    
                    metrics_text = f"<i>Confidence: {confidence:.2f} | Sources: {sources} | Duration: {duration:.1f}s</i>"
                    story.append(Paragraph(metrics_text, self.styles['Metadata']))
                    
                    # Reasoning steps
                    if 'reasoning_steps' in metadata:
                        steps_text = "<i>Reasoning Steps:</i>"
                        story.append(Paragraph(steps_text, self.styles['Metadata']))
                        
                        for j, step in enumerate(metadata['reasoning_steps'], 1):
                            stage = step.get('stage', '').replace('_', ' ').title()
                            step_duration = step.get('duration', 0)
                            success = step.get('success', False)
                            status = "✅" if success else "❌"
                            
                            step_text = f"<i>{j}. {stage} ({step_duration:.2f}s) {status}</i>"
                            story.append(Paragraph(step_text, self.styles['Metadata']))
                
                story.append(Spacer(1, 18))
            
            # Build PDF
            try:
                doc.build(story)
            except Exception as build_error:
                logger.error(f"PDF build failed: {build_error}")
                # Try with simplified content
                simplified_story = []
                simplified_story.append(Paragraph("Deep Research Agent - Chat Session", self.styles['Title']))
                simplified_story.append(Spacer(1, 12))
                simplified_story.append(Paragraph(f"Session ID: {session_id}", self.styles['Normal']))
                simplified_story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
                simplified_story.append(Paragraph(f"Total Messages: {len(chat_history)}", self.styles['Normal']))
                simplified_story.append(Spacer(1, 12))
                simplified_story.append(Paragraph("Chat History", self.styles['Heading2']))
                simplified_story.append(Spacer(1, 12))
                
                for i, message in enumerate(chat_history, 1):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    
                    # Simple plain text version
                    plain_content = str(content).replace('<', '').replace('>', '').replace('&', '')
                    if len(plain_content) > 500:
                        plain_content = plain_content[:500] + "... (truncated)"
                    
                    header = f"Message {i} ({role})"
                    simplified_story.append(Paragraph(header, self.styles['Heading3']))
                    simplified_story.append(Paragraph(plain_content, self.styles['Normal']))
                    simplified_story.append(Spacer(1, 12))
                
                doc.build(simplified_story)
            
            # Save to file if path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(buffer.getvalue())
            
            # Return buffer
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise Exception(f"Failed to export chat as PDF: {e}")
    
    def _clean_content_for_pdf(self, content: str) -> str:
        """Clean content for PDF rendering."""
        try:
            import re
            from html import unescape
            
            # First, strip any existing HTML tags completely to avoid malformed XML
            # This removes all HTML tags and leaves just the text content
            content = re.sub(r'<[^>]+>', '', content)
            
            # Unescape any HTML entities that might remain
            content = unescape(content)
            
            # Now escape characters that would break XML
            content = content.replace('&', '&amp;')
            content = content.replace('<', '&lt;')
            content = content.replace('>', '&gt;')
            
            # Handle markdown-like formatting with proper opening and closing tags
            # Replace **text** with <b>text</b> (non-greedy)
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content, flags=re.DOTALL)
            
            # Replace __text__ with <i>text</i> (non-greedy)
            content = re.sub(r'__(.*?)__', r'<i>\1</i>', content, flags=re.DOTALL)
            
            # Replace *text* with <i>text</i> (non-greedy, but avoid double conversion)
            content = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', content)
            
            # Handle line breaks - convert multiple newlines to paragraph breaks
            content = re.sub(r'\n\s*\n', '<br/><br/>', content)
            content = content.replace('\n', '<br/>')
            
            # Clean up any malformed or nested tags that might have been created
            # Remove empty tags
            content = re.sub(r'<b>\s*</b>', '', content)
            content = re.sub(r'<i>\s*</i>', '', content)
            
            # Fix nested tags by flattening them
            content = re.sub(r'<b>([^<]*)<b>', r'<b>\1', content)  # Remove nested bold opening
            content = re.sub(r'</b>([^>]*)</b>', r'\1</b>', content)  # Remove nested bold closing
            content = re.sub(r'<i>([^<]*)<i>', r'<i>\1', content)  # Remove nested italic opening
            content = re.sub(r'</i>([^>]*)</i>', r'\1</i>', content)  # Remove nested italic closing
            
            # Ensure all bold tags are properly closed
            open_bold = content.count('<b>') - content.count('</b>')
            if open_bold > 0:
                content += '</b>' * open_bold
            elif open_bold < 0:
                # Remove excess closing tags
                for _ in range(abs(open_bold)):
                    content = content.replace('</b>', '', 1)
            
            # Ensure all italic tags are properly closed
            open_italic = content.count('<i>') - content.count('</i>')
            if open_italic > 0:
                content += '</i>' * open_italic
            elif open_italic < 0:
                # Remove excess closing tags
                for _ in range(abs(open_italic)):
                    content = content.replace('</i>', '', 1)
            
            # Final cleanup - remove any remaining malformed patterns
            content = re.sub(r'<b>(?!.*?</b>).*?$', lambda m: m.group(0).replace('<b>', ''), content)
            content = re.sub(r'<i>(?!.*?</i>).*?$', lambda m: m.group(0).replace('<i>', ''), content)
            
            # Limit content length to prevent ReportLab issues
            if len(content) > 8000:
                content = content[:8000] + "... (content truncated for PDF export)"
            
            return content
            
        except Exception as e:
            logger.error(f"Content cleaning failed: {e}")
            # Return a safe, plain text version as fallback
            try:
                import re
                safe_content = str(content)
                # Strip all HTML/XML tags
                safe_content = re.sub(r'<[^>]+>', '', safe_content)
                # Escape remaining special characters
                safe_content = safe_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Limit length
                if len(safe_content) > 2000:
                    safe_content = safe_content[:2000] + "... (content simplified for PDF export)"
                return safe_content
            except:
                return "Content could not be processed for PDF export."
    
    def export_research_summary(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        reasoning_steps: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """
        Export individual research summary.
        
        Args:
            query: Research query
            answer: Generated answer
            sources: List of source documents
            reasoning_steps: List of reasoning steps
            metadata: Additional metadata
            format_type: Export format ('markdown' or 'pdf')
        
        Returns:
            Formatted content string
        """
        try:
            if format_type.lower() == "markdown":
                return self._create_research_markdown(query, answer, sources, reasoning_steps, metadata)
            elif format_type.lower() == "pdf":
                # Create a single-item chat history for PDF export
                chat_history = [
                    {"role": "user", "content": query, "timestamp": datetime.now().isoformat()},
                    {
                        "role": "assistant", 
                        "content": answer, 
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "sources": sources,
                            "reasoning_steps": reasoning_steps,
                            **metadata
                        }
                    }
                ]
                
                buffer = self.export_chat_as_pdf(chat_history, f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                return buffer.getvalue()
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Research summary export failed: {e}")
            raise Exception(f"Failed to export research summary: {e}")
    
    def _create_research_markdown(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        reasoning_steps: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """Create markdown format for research summary."""
        content = []
        
        content.append("# 🔍 Research Summary")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        content.append("## ❓ Research Query")
        content.append("")
        content.append(query)
        content.append("")
        
        content.append("## 💡 Answer")
        content.append("")
        content.append(answer)
        content.append("")
        
        if sources:
            content.append("## 📚 Sources")
            content.append("")
            for i, source in enumerate(sources, 1):
                content.append(f"{i}. **{source.get('filename', 'Unknown')}** (Relevance: {source.get('relevance_score', 'N/A')})")
                content.append(f"   *{source.get('content', '')[:200]}...*")
                content.append("")
        
        if reasoning_steps:
            content.append("## 🧠 Reasoning Process")
            content.append("")
            for i, step in enumerate(reasoning_steps, 1):
                stage = step.get('stage', '').replace('_', ' ').title()
                duration = step.get('duration', 0)
                success = step.get('success', False)
                status = "✅" if success else "❌"
                
                content.append(f"{i}. **{stage}** ({duration:.2f}s) {status}")
                
                if 'details' in step:
                    content.append(f"   {step['details']}")
                content.append("")
        
        return "\n".join(content)