"""
Research Report Generation System
Generates comprehensive reports from research findings with multiple export formats
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

# Report generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import markdown
from fpdf import FPDF

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from config import ResearchConfig
from research_agent import ResearchResult, ResearchFinding


class ReportSection(BaseModel):
    """Individual report section"""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    subsections: List[Dict[str, str]] = Field(description="Subsections with title and content")
    citations: List[str] = Field(description="Citations for this section")
    confidence_level: float = Field(description="Confidence level for this section")


class ReportStructure(BaseModel):
    """Complete report structure"""
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Executive summary")
    introduction: str = Field(description="Introduction section")
    methodology: str = Field(description="Research methodology")
    sections: List[ReportSection] = Field(description="Main report sections")
    conclusions: str = Field(description="Conclusions section")
    recommendations: List[str] = Field(description="Recommendations")
    limitations: str = Field(description="Research limitations")
    references: List[str] = Field(description="List of all references")


@dataclass
class ReportMetadata:
    """Report metadata"""
    title: str
    author: str
    created_date: str
    research_query: str
    total_sources: int
    confidence_score: float
    format_type: str
    file_path: str


class ReportGenerator:
    """Advanced report generator with multiple format support"""
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.config.create_directories()
        
        # Initialize LLM for report generation
        model_config = self.config.get_ollama_config()
        self.llm = ChatOllama(**model_config["summarization"])
        
        # Initialize parsers
        self.structure_parser = PydanticOutputParser(pydantic_object=ReportStructure)
        
        # Report templates
        self.templates = {
            "academic": self._get_academic_template(),
            "business": self._get_business_template(),
            "executive": self._get_executive_template(),
            "detailed": self._get_detailed_template()
        }
    
    def _get_academic_template(self) -> str:
        """Academic report template"""
        return """
        Create an academic-style research report with the following structure:
        
        1. **Title**: Clear, descriptive title reflecting the research scope
        2. **Executive Summary**: Concise overview of key findings (200-300 words)
        3. **Introduction**: Background, research question, and objectives
        4. **Methodology**: Brief description of research approach and data sources
        5. **Findings**: Detailed analysis organized by themes or categories
        6. **Discussion**: Interpretation of findings and their significance
        7. **Conclusions**: Summary of main insights and their implications
        8. **Limitations**: Acknowledgment of research limitations and constraints
        9. **References**: Comprehensive list of all sources cited
        
        Use formal academic language, maintain objectivity, and support all claims with evidence.
        """
    
    def _get_business_template(self) -> str:
        """Business report template"""
        return """
        Create a business-focused research report with the following structure:
        
        1. **Executive Summary**: Key findings and business implications (150-200 words)
        2. **Background**: Context and business problem being addressed
        3. **Key Findings**: Main insights organized by business impact
        4. **Analysis**: Detailed examination of findings with business context
        5. **Recommendations**: Actionable business recommendations
        6. **Implementation Considerations**: Practical aspects of applying findings
        7. **Risk Assessment**: Potential risks and mitigation strategies
        8. **Conclusion**: Summary and next steps
        9. **Appendices**: Supporting data and sources
        
        Use clear, business-oriented language and focus on actionable insights.
        """
    
    def _get_executive_template(self) -> str:
        """Executive summary template"""
        return """
        Create a concise executive report with the following structure:
        
        1. **Executive Summary**: Comprehensive overview (300-400 words)
        2. **Key Insights**: Top 5-7 most important findings
        3. **Strategic Implications**: What this means for decision-making
        4. **Recommendations**: Top 3-5 actionable recommendations
        5. **Next Steps**: Suggested follow-up actions
        6. **Supporting Evidence**: Brief overview of evidence base
        
        Use executive-level language, focus on strategic implications, and keep it concise.
        """
    
    def _get_detailed_template(self) -> str:
        """Detailed analysis template"""
        return """
        Create a comprehensive detailed research report with the following structure:
        
        1. **Executive Summary**: Overview of entire research
        2. **Introduction**: Detailed background and context
        3. **Research Framework**: Methodology and approach
        4. **Detailed Findings**: Comprehensive analysis with multiple sections
        5. **Evidence Analysis**: Critical examination of supporting evidence
        6. **Cross-Analysis**: Relationships and patterns across findings
        7. **Quality Assessment**: Evaluation of research quality and reliability
        8. **Implications**: Broader implications and significance
        9. **Recommendations**: Detailed, actionable recommendations
        10. **Future Research**: Suggestions for additional research
        11. **Conclusion**: Comprehensive summary
        12. **References and Sources**: Complete bibliography
        
        Provide thorough analysis with extensive detail and multiple perspectives.
        """
    
    async def generate_report_structure(self, 
                                      research_result: ResearchResult,
                                      template_type: str = "detailed",
                                      custom_requirements: str = None) -> ReportStructure:
        """Generate structured report content"""
        try:
            template = self.templates.get(template_type, self.templates["detailed"])
            
            structure_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert research report writer. Generate a comprehensive, well-structured research report based on the provided research findings.
                
                Template to follow:
                {template}
                
                Additional requirements: {custom_requirements or "Follow standard academic conventions"}
                
                Guidelines:
                - Create clear, logical flow between sections
                - Support all claims with evidence from the findings
                - Maintain appropriate tone and style for the report type
                - Include proper citations and references
                - Ensure conclusions are well-supported by evidence
                - Identify and acknowledge limitations
                
                {self.structure_parser.get_format_instructions()}"""),
                ("human", """
                Research Query: {query}
                Research Findings Summary:
                - Total Findings: {total_findings}
                - Confidence Score: {confidence_score}
                - Sources Used: {sources_count}
                
                Key Findings:
                {findings_text}
                
                Research Synthesis:
                {synthesis}
                
                Please generate a comprehensive report structure based on this research.
                """)
            ])
            
            # Prepare findings text
            top_findings = sorted(research_result.findings, key=lambda x: x.relevance_score, reverse=True)[:15]
            findings_text = "\n\n".join([
                f"Finding {i+1} (Relevance: {f.relevance_score:.2f}, Source: {f.source}):\n{f.content[:400]}..."
                for i, f in enumerate(top_findings)
            ])
            
            response = await self.llm.ainvoke(
                structure_prompt.format_messages(
                    query=research_result.query.original_query,
                    total_findings=len(research_result.findings),
                    confidence_score=research_result.confidence_score,
                    sources_count=len(research_result.sources_used),
                    findings_text=findings_text,
                    synthesis=research_result.synthesis[:1000] + "..." if len(research_result.synthesis) > 1000 else research_result.synthesis
                )
            )
            
            report_structure = self.structure_parser.parse(response.content)
            return report_structure
            
        except Exception as e:
            print(f"Error generating report structure: {e}")
            # Return a basic structure as fallback
            return ReportStructure(
                title=f"Research Report: {research_result.query.original_query}",
                executive_summary="Research synthesis not available due to processing error.",
                introduction="This report presents findings from automated research.",
                methodology="Automated research using semantic search and analysis.",
                sections=[],
                conclusions="Unable to generate conclusions due to processing error.",
                recommendations=[],
                limitations="Report generation encountered technical limitations.",
                references=research_result.sources_used
            )
    
    def _create_markdown_report(self, structure: ReportStructure, metadata: ReportMetadata) -> str:
        """Generate Markdown format report"""
        md_content = []
        
        # Title and metadata
        md_content.append(f"# {structure.title}\n")
        md_content.append(f"**Author:** {metadata.author}")
        md_content.append(f"**Date:** {metadata.created_date}")
        md_content.append(f"**Research Query:** {metadata.research_query}")
        md_content.append(f"**Confidence Score:** {metadata.confidence_score:.2f}")
        md_content.append(f"**Total Sources:** {metadata.total_sources}\n")
        md_content.append("---\n")
        
        # Executive Summary
        md_content.append("## Executive Summary\n")
        md_content.append(f"{structure.executive_summary}\n")
        
        # Table of Contents
        md_content.append("## Table of Contents\n")
        md_content.append("1. [Introduction](#introduction)")
        md_content.append("2. [Methodology](#methodology)")
        for i, section in enumerate(structure.sections, 3):
            section_id = section.title.lower().replace(" ", "-").replace(",", "")
            md_content.append(f"{i}. [{section.title}](#{section_id})")
        md_content.append(f"{len(structure.sections) + 3}. [Conclusions](#conclusions)")
        md_content.append(f"{len(structure.sections) + 4}. [Recommendations](#recommendations)")
        md_content.append(f"{len(structure.sections) + 5}. [Limitations](#limitations)")
        md_content.append(f"{len(structure.sections) + 6}. [References](#references)\n")
        
        # Introduction
        md_content.append("## Introduction\n")
        md_content.append(f"{structure.introduction}\n")
        
        # Methodology
        md_content.append("## Methodology\n")
        md_content.append(f"{structure.methodology}\n")
        
        # Main sections
        for section in structure.sections:
            section_id = section.title.lower().replace(" ", "-").replace(",", "")
            md_content.append(f"## {section.title}\n")
            md_content.append(f"{section.content}\n")
            
            # Subsections
            for subsection in section.subsections:
                md_content.append(f"### {subsection.get('title', 'Subsection')}\n")
                md_content.append(f"{subsection.get('content', '')}\n")
            
            # Section citations
            if section.citations:
                md_content.append("**Sources:**")
                for citation in section.citations:
                    md_content.append(f"- {citation}")
                md_content.append("")
        
        # Conclusions
        md_content.append("## Conclusions\n")
        md_content.append(f"{structure.conclusions}\n")
        
        # Recommendations
        md_content.append("## Recommendations\n")
        for i, rec in enumerate(structure.recommendations, 1):
            md_content.append(f"{i}. {rec}")
        md_content.append("")
        
        # Limitations
        md_content.append("## Limitations\n")
        md_content.append(f"{structure.limitations}\n")
        
        # References
        md_content.append("## References\n")
        for i, ref in enumerate(structure.references, 1):
            md_content.append(f"{i}. {ref}")
        
        return "\n".join(md_content)
    
    def _create_html_report(self, structure: ReportStructure, metadata: ReportMetadata) -> str:
        """Generate HTML format report"""
        # Convert markdown to HTML
        markdown_content = self._create_markdown_report(structure, metadata)
        
        # Custom CSS for professional styling
        css_style = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }
            h3 {
                color: #7f8c8d;
            }
            .metadata {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .confidence-score {
                font-weight: bold;
                color: #27ae60;
            }
            .section {
                margin-bottom: 30px;
            }
            .citation {
                font-style: italic;
                color: #7f8c8d;
                margin-left: 20px;
            }
            ol, ul {
                padding-left: 20px;
            }
            blockquote {
                border-left: 4px solid #3498db;
                margin: 0;
                padding-left: 20px;
                background-color: #f8f9fa;
            }
        </style>
        """
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['toc', 'tables', 'fenced_code'])
        
        # Wrap in complete HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{structure.title}</title>
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        return full_html
    
    def _create_pdf_report(self, structure: ReportStructure, metadata: ReportMetadata, file_path: str):
        """Generate PDF format report using ReportLab"""
        try:
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkslategray
            )
            
            story = []
            
            # Title
            story.append(Paragraph(structure.title, title_style))
            story.append(Spacer(1, 12))
            
            # Metadata table
            metadata_data = [
                ['Author:', metadata.author],
                ['Date:', metadata.created_date],
                ['Research Query:', metadata.research_query],
                ['Confidence Score:', f"{metadata.confidence_score:.2f}"],
                ['Total Sources:', str(metadata.total_sources)]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(structure.executive_summary, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Introduction
            story.append(Paragraph("Introduction", heading_style))
            story.append(Paragraph(structure.introduction, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Methodology
            story.append(Paragraph("Methodology", heading_style))
            story.append(Paragraph(structure.methodology, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Main sections
            for section in structure.sections:
                story.append(Paragraph(section.title, heading_style))
                story.append(Paragraph(section.content, styles['Normal']))
                
                for subsection in section.subsections:
                    if subsection.get('title') and subsection.get('content'):
                        story.append(Paragraph(subsection['title'], styles['Heading2']))
                        story.append(Paragraph(subsection['content'], styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            # Conclusions
            story.append(Paragraph("Conclusions", heading_style))
            story.append(Paragraph(structure.conclusions, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            for i, rec in enumerate(structure.recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Limitations
            story.append(Paragraph("Limitations", heading_style))
            story.append(Paragraph(structure.limitations, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # References
            story.append(Paragraph("References", heading_style))
            for i, ref in enumerate(structure.references, 1):
                story.append(Paragraph(f"{i}. {ref}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
        except Exception as e:
            print(f"Error creating PDF: {e}")
            # Fallback to simple PDF using FPDF
            self._create_simple_pdf_report(structure, metadata, file_path)
    
    def _create_simple_pdf_report(self, structure: ReportStructure, metadata: ReportMetadata, file_path: str):
        """Create simple PDF using FPDF as fallback"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Title
            pdf.cell(0, 10, structure.title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
            pdf.ln(10)
            
            # Metadata
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Report Metadata', ln=True)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'Author: {metadata.author}', ln=True)
            pdf.cell(0, 6, f'Date: {metadata.created_date}', ln=True)
            pdf.cell(0, 6, f'Confidence Score: {metadata.confidence_score:.2f}', ln=True)
            pdf.ln(5)
            
            # Executive Summary
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Executive Summary', ln=True)
            pdf.set_font('Arial', '', 10)
            
            # Split text for proper formatting
            exec_summary = structure.executive_summary.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, exec_summary)
            pdf.ln(5)
            
            # Save PDF
            pdf.output(file_path)
            
        except Exception as e:
            print(f"Error creating simple PDF: {e}")
    
    async def generate_report(self, 
                            research_result: ResearchResult,
                            format_type: str = "markdown",
                            template_type: str = "detailed",
                            custom_requirements: str = None,
                            output_filename: str = None) -> ReportMetadata:
        """Generate a complete research report"""
        try:
            # Generate report structure
            structure = await self.generate_report_structure(
                research_result, 
                template_type, 
                custom_requirements
            )
            
            # Create metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_filename or f"research_report_{timestamp}"
            
            metadata = ReportMetadata(
                title=structure.title,
                author="Deep Research Agent",
                created_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                research_query=research_result.query.original_query,
                total_sources=len(research_result.sources_used),
                confidence_score=research_result.confidence_score,
                format_type=format_type,
                file_path=""
            )
            
            # Generate report in requested format
            if format_type.lower() == "markdown":
                content = self._create_markdown_report(structure, metadata)
                file_path = self.config.REPORTS_DIR / f"{filename}.md"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            elif format_type.lower() == "html":
                content = self._create_html_report(structure, metadata)
                file_path = self.config.REPORTS_DIR / f"{filename}.html"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            elif format_type.lower() == "pdf":
                file_path = self.config.REPORTS_DIR / f"{filename}.pdf"
                self._create_pdf_report(structure, metadata, str(file_path))
                
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            metadata.file_path = str(file_path)
            
            # Save metadata
            metadata_file = self.config.REPORTS_DIR / f"{filename}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            print(f"Report generated successfully: {file_path}")
            return metadata
            
        except Exception as e:
            print(f"Error generating report: {e}")
            raise
    
    async def generate_multiple_formats(self, 
                                      research_result: ResearchResult,
                                      template_type: str = "detailed",
                                      output_filename: str = None) -> List[ReportMetadata]:
        """Generate report in multiple formats"""
        formats = ["markdown", "html", "pdf"]
        metadata_list = []
        
        for format_type in formats:
            try:
                metadata = await self.generate_report(
                    research_result,
                    format_type=format_type,
                    template_type=template_type,
                    output_filename=output_filename
                )
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Failed to generate {format_type} format: {e}")
        
        return metadata_list
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports"""
        reports = []
        
        for metadata_file in self.config.REPORTS_DIR.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    reports.append(metadata)
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
        
        return sorted(reports, key=lambda x: x.get('created_date', ''), reverse=True)


# Usage example
if __name__ == "__main__":
    async def main():
        from embedding_system import AdvancedEmbeddingSystem
        from research_workflow import ResearchWorkflowOrchestrator
        
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add documents
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Document loading: {result}")
        
        if result.get("status") == "success":
            # Create orchestrator and run research
            orchestrator = ResearchWorkflowOrchestrator(embedding_system, config)
            
            research_result = await orchestrator.orchestrate_research(
                "What are the most important factors for restaurant success based on customer feedback?"
            )
            
            if research_result['status'] == 'success':
                # Generate report
                report_generator = ReportGenerator(config)
                
                # Generate in multiple formats
                metadata_list = await report_generator.generate_multiple_formats(
                    research_result['result'],
                    template_type="business",
                    output_filename="restaurant_success_factors"
                )
                
                print(f"\nGenerated {len(metadata_list)} report formats:")
                for metadata in metadata_list:
                    print(f"- {metadata.format_type.upper()}: {metadata.file_path}")
                
                # List all reports
                all_reports = report_generator.list_reports()
                print(f"\nTotal reports available: {len(all_reports)}")
    
    # Run the async function
    asyncio.run(main())