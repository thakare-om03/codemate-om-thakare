"""
Quality Control and Evaluation System
Implements comprehensive quality assessment for research findings and reports
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import re
from statistics import mean, stdev
from collections import defaultdict

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# Local imports
from config import ResearchConfig
from research_agent import ResearchResult, ResearchFinding
from report_generator import ReportStructure


class QualityMetrics(BaseModel):
    """Quality assessment metrics"""
    relevance_score: float = Field(description="Relevance to original query")
    accuracy_score: float = Field(description="Factual accuracy assessment")
    completeness_score: float = Field(description="Coverage completeness")
    coherence_score: float = Field(description="Logical coherence")
    source_reliability: float = Field(description="Source reliability rating")
    evidence_strength: float = Field(description="Strength of supporting evidence")
    bias_score: float = Field(description="Bias assessment (lower is better)")
    recency_score: float = Field(description="Information recency")
    overall_quality: float = Field(description="Overall quality score")


class SourceReliability(BaseModel):
    """Source reliability assessment"""
    source_name: str = Field(description="Name of the source")
    domain_authority: float = Field(description="Domain authority score")
    content_quality: float = Field(description="Content quality assessment")
    citation_count: int = Field(description="Number of citations")
    peer_review_status: bool = Field(description="Whether peer-reviewed")
    publication_date: Optional[str] = Field(description="Publication date")
    reliability_score: float = Field(description="Overall reliability score")
    reliability_reasoning: str = Field(description="Reasoning for reliability score")


class FindingQuality(BaseModel):
    """Individual finding quality assessment"""
    finding_id: str = Field(description="Unique finding identifier")
    content_summary: str = Field(description="Brief content summary")
    metrics: QualityMetrics = Field(description="Quality metrics")
    confidence_interval: Tuple[float, float] = Field(description="Confidence interval")
    quality_issues: List[str] = Field(description="Identified quality issues")
    improvement_suggestions: List[str] = Field(description="Suggestions for improvement")


class ResearchQualityReport(BaseModel):
    """Comprehensive quality assessment report"""
    query_analysis: str = Field(description="Analysis of original query quality")
    overall_metrics: QualityMetrics = Field(description="Overall research quality")
    finding_assessments: List[FindingQuality] = Field(description="Individual finding assessments")
    source_reliability: List[SourceReliability] = Field(description="Source reliability assessments")
    methodological_assessment: str = Field(description="Assessment of research methodology")
    limitations_identified: List[str] = Field(description="Research limitations")
    recommendations: List[str] = Field(description="Quality improvement recommendations")
    quality_grade: str = Field(description="Overall quality grade (A-F)")


@dataclass
class EvaluationConfig:
    """Configuration for quality evaluation"""
    min_confidence_threshold: float = 0.7
    min_source_reliability: float = 0.6
    bias_detection_threshold: float = 0.3
    recency_weight: float = 0.15
    source_diversity_weight: float = 0.2
    evidence_consistency_weight: float = 0.25
    completeness_weight: float = 0.2
    accuracy_weight: float = 0.2


class QualityController:
    """Advanced quality control and evaluation system"""
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.eval_config = EvaluationConfig()
        
        # Initialize LLM for quality assessment
        model_config = self.config.get_ollama_config()
        self.llm = ChatOllama(**model_config["reasoning"])
        
        # Initialize parsers
        self.quality_parser = PydanticOutputParser(pydantic_object=QualityMetrics)
        self.reliability_parser = PydanticOutputParser(pydantic_object=SourceReliability)
        self.report_parser = PydanticOutputParser(pydantic_object=ResearchQualityReport)
        
        # Quality assessment prompts
        self.quality_prompts = self._initialize_quality_prompts()
        
        # Known reliable domains and patterns
        self.reliable_domains = {
            'academic': ['edu', 'scholar.google', 'pubmed', 'arxiv', 'ieee', 'acm'],
            'government': ['gov', 'who.int', 'cdc.gov', 'fda.gov'],
            'reputable_news': ['reuters.com', 'bbc.com', 'npr.org', 'apnews.com'],
            'professional': ['harvard.edu', 'mit.edu', 'stanford.edu', 'nature.com', 'science.org']
        }
        
        # Bias indicators
        self.bias_indicators = [
            'obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely',
            'all experts agree', 'everyone knows', 'studies prove',
            'the only way', 'always', 'never', 'completely'
        ]
    
    def _initialize_quality_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Initialize quality assessment prompts"""
        return {
            'relevance': ChatPromptTemplate.from_messages([
                ("system", """You are an expert evaluator assessing the relevance of research findings to the original query.
                
                Evaluate how well each finding addresses the research question:
                - Direct relevance: Does it directly answer the question?
                - Contextual relevance: Does it provide important context?
                - Tangential relevance: Is it only loosely related?
                
                Score from 0.0 (irrelevant) to 1.0 (highly relevant).
                {format_instructions}"""),
                ("human", "Original Query: {query}\n\nFinding: {finding}\n\nAssess relevance:")
            ]),
            
            'accuracy': ChatPromptTemplate.from_messages([
                ("system", """You are a fact-checking expert evaluating the accuracy of research findings.
                
                Assess factual accuracy considering:
                - Verifiable claims and data
                - Consistency with established knowledge
                - Presence of supporting evidence
                - Detection of potential misinformation
                
                Score from 0.0 (inaccurate) to 1.0 (highly accurate).
                {format_instructions}"""),
                ("human", "Finding to evaluate: {finding}\n\nSource: {source}\n\nAssess accuracy:")
            ]),
            
            'source_reliability': ChatPromptTemplate.from_messages([
                ("system", """You are an expert in evaluating source credibility and reliability.
                
                Assess source reliability based on:
                - Domain authority and reputation
                - Author expertise and credentials
                - Publication standards and peer review
                - Citation practices and references
                - Potential conflicts of interest
                
                {format_instructions}"""),
                ("human", "Source to evaluate: {source}\n\nContent context: {content}\n\nAssess reliability:")
            ]),
            
            'bias_detection': ChatPromptTemplate.from_messages([
                ("system", """You are an expert in detecting bias and evaluating objectivity.
                
                Identify potential biases:
                - Selection bias in evidence presentation
                - Confirmation bias in interpretation
                - Language that suggests predetermined conclusions
                - Missing counterarguments or alternative viewpoints
                - Emotional or loaded language
                
                Score bias from 0.0 (unbiased) to 1.0 (highly biased).
                {format_instructions}"""),
                ("human", "Content to analyze: {content}\n\nDetect bias and assess objectivity:")
            ])
        }
    
    async def assess_finding_quality(self, 
                                   finding: ResearchFinding, 
                                   original_query: str) -> FindingQuality:
        """Assess quality of individual research finding"""
        try:
            # Assess relevance
            relevance_prompt = self.quality_prompts['relevance']
            relevance_response = await self.llm.ainvoke(
                relevance_prompt.format_messages(
                    query=original_query,
                    finding=finding.content,
                    format_instructions=self.quality_parser.get_format_instructions()
                )
            )
            
            # Parse relevance or use fallback
            try:
                relevance_metrics = self.quality_parser.parse(relevance_response.content)
                relevance_score = relevance_metrics.relevance_score
            except:
                relevance_score = finding.relevance_score  # Use existing score as fallback
            
            # Assess accuracy
            accuracy_score = await self._assess_accuracy(finding)
            
            # Assess source reliability
            source_reliability = await self._assess_source_reliability(finding.source, finding.content)
            
            # Detect bias
            bias_score = await self._detect_bias(finding.content)
            
            # Calculate additional metrics
            completeness_score = self._assess_completeness(finding.content, original_query)
            coherence_score = self._assess_coherence(finding.content)
            evidence_strength = self._assess_evidence_strength(finding.content)
            recency_score = self._assess_recency(finding)
            
            # Compile quality metrics
            metrics = QualityMetrics(
                relevance_score=relevance_score,
                accuracy_score=accuracy_score,
                completeness_score=completeness_score,
                coherence_score=coherence_score,
                source_reliability=source_reliability.reliability_score,
                evidence_strength=evidence_strength,
                bias_score=bias_score,
                recency_score=recency_score,
                overall_quality=self._calculate_overall_quality({
                    'relevance': relevance_score,
                    'accuracy': accuracy_score,
                    'completeness': completeness_score,
                    'coherence': coherence_score,
                    'reliability': source_reliability.reliability_score,
                    'evidence': evidence_strength,
                    'bias': 1.0 - bias_score,  # Invert bias score
                    'recency': recency_score
                })
            )
            
            # Identify quality issues
            quality_issues = self._identify_quality_issues(metrics, finding)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(metrics, quality_issues)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(metrics)
            
            return FindingQuality(
                finding_id=f"finding_{hash(finding.content) % 10000}",
                content_summary=finding.content[:200] + "..." if len(finding.content) > 200 else finding.content,
                metrics=metrics,
                confidence_interval=confidence_interval,
                quality_issues=quality_issues,
                improvement_suggestions=improvement_suggestions
            )
            
        except Exception as e:
            print(f"Error assessing finding quality: {e}")
            # Return basic assessment as fallback
            return FindingQuality(
                finding_id=f"finding_{hash(finding.content) % 10000}",
                content_summary=finding.content[:200] + "...",
                metrics=QualityMetrics(
                    relevance_score=finding.relevance_score,
                    accuracy_score=0.5,
                    completeness_score=0.5,
                    coherence_score=0.5,
                    source_reliability=0.5,
                    evidence_strength=0.5,
                    bias_score=0.3,
                    recency_score=0.5,
                    overall_quality=0.5
                ),
                confidence_interval=(0.3, 0.7),
                quality_issues=["Assessment error occurred"],
                improvement_suggestions=["Manual review recommended"]
            )
    
    async def _assess_accuracy(self, finding: ResearchFinding) -> float:
        """Assess factual accuracy of a finding"""
        try:
            accuracy_prompt = self.quality_prompts['accuracy']
            response = await self.llm.ainvoke(
                accuracy_prompt.format_messages(
                    finding=finding.content,
                    source=finding.source,
                    format_instructions=self.quality_parser.get_format_instructions()
                )
            )
            
            metrics = self.quality_parser.parse(response.content)
            return metrics.accuracy_score
        except:
            # Fallback accuracy assessment based on content analysis
            return self._heuristic_accuracy_assessment(finding.content)
    
    def _heuristic_accuracy_assessment(self, content: str) -> float:
        """Heuristic accuracy assessment based on content patterns"""
        score = 0.5  # Base score
        
        # Positive indicators
        if re.search(r'\d+%|\d+\.\d+', content):  # Contains statistics
            score += 0.1
        if 'study' in content.lower() or 'research' in content.lower():
            score += 0.1
        if 'according to' in content.lower() or 'found that' in content.lower():
            score += 0.1
        
        # Negative indicators
        if any(word in content.lower() for word in ['might', 'could', 'possibly', 'perhaps']):
            score -= 0.1
        if any(indicator in content.lower() for indicator in self.bias_indicators):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _assess_source_reliability(self, source: str, content: str) -> SourceReliability:
        """Assess source reliability"""
        try:
            reliability_prompt = self.quality_prompts['source_reliability']
            response = await self.llm.ainvoke(
                reliability_prompt.format_messages(
                    source=source,
                    content=content[:500],  # Limit content for context
                    format_instructions=self.reliability_parser.get_format_instructions()
                )
            )
            
            return self.reliability_parser.parse(response.content)
        except:
            # Fallback reliability assessment
            return self._heuristic_reliability_assessment(source, content)
    
    def _heuristic_reliability_assessment(self, source: str, content: str) -> SourceReliability:
        """Heuristic source reliability assessment"""
        domain_authority = 0.5
        
        # Check against known reliable domains
        source_lower = source.lower()
        for category, domains in self.reliable_domains.items():
            if any(domain in source_lower for domain in domains):
                if category == 'academic':
                    domain_authority = 0.9
                elif category == 'government':
                    domain_authority = 0.85
                elif category == 'professional':
                    domain_authority = 0.8
                elif category == 'reputable_news':
                    domain_authority = 0.7
                break
        
        # Content quality assessment
        content_quality = 0.5
        if len(content) > 200:  # Substantial content
            content_quality += 0.1
        if re.search(r'references?|citations?|bibliography', content.lower()):
            content_quality += 0.2
        
        reliability_score = (domain_authority + content_quality) / 2
        
        return SourceReliability(
            source_name=source,
            domain_authority=domain_authority,
            content_quality=content_quality,
            citation_count=0,  # Unknown
            peer_review_status=domain_authority > 0.8,
            publication_date=None,
            reliability_score=reliability_score,
            reliability_reasoning=f"Assessed based on domain reputation and content indicators"
        )
    
    async def _detect_bias(self, content: str) -> float:
        """Detect bias in content"""
        try:
            bias_prompt = self.quality_prompts['bias_detection']
            response = await self.llm.ainvoke(
                bias_prompt.format_messages(
                    content=content,
                    format_instructions=self.quality_parser.get_format_instructions()
                )
            )
            
            metrics = self.quality_parser.parse(response.content)
            return metrics.bias_score
        except:
            # Fallback bias detection
            return self._heuristic_bias_detection(content)
    
    def _heuristic_bias_detection(self, content: str) -> float:
        """Heuristic bias detection"""
        bias_score = 0.0
        content_lower = content.lower()
        
        # Check for bias indicators
        bias_count = sum(1 for indicator in self.bias_indicators if indicator in content_lower)
        bias_score += min(0.3, bias_count * 0.05)
        
        # Check for emotional language
        emotional_words = ['terrible', 'amazing', 'shocking', 'unbelievable', 'devastating', 'incredible']
        emotional_count = sum(1 for word in emotional_words if word in content_lower)
        bias_score += min(0.2, emotional_count * 0.05)
        
        # Check for one-sided presentation
        if not any(word in content_lower for word in ['however', 'although', 'despite', 'on the other hand']):
            bias_score += 0.1
        
        return min(1.0, bias_score)
    
    def _assess_completeness(self, content: str, query: str) -> float:
        """Assess completeness of content relative to query"""
        # Simple heuristic based on content length and query coverage
        base_score = min(1.0, len(content) / 500)  # Normalize to 500 chars
        
        # Check if content addresses key query terms
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        coverage = len(query_words.intersection(content_words)) / len(query_words)
        
        return (base_score + coverage) / 2
    
    def _assess_coherence(self, content: str) -> float:
        """Assess logical coherence of content"""
        sentences = content.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence indicators
        coherence_indicators = ['therefore', 'however', 'furthermore', 'additionally', 'consequently']
        indicator_count = sum(1 for sentence in sentences 
                            for indicator in coherence_indicators 
                            if indicator in sentence.lower())
        
        # Normalize by sentence count
        coherence_score = min(1.0, 0.5 + (indicator_count / len(sentences)))
        return coherence_score
    
    def _assess_evidence_strength(self, content: str) -> float:
        """Assess strength of evidence in content"""
        evidence_indicators = [
            'study', 'research', 'data', 'evidence', 'findings',
            'results', 'analysis', 'survey', 'experiment', 'trial'
        ]
        
        content_lower = content.lower()
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content_lower)
        
        # Check for specific evidence types
        if re.search(r'\d+%|\d+\.\d+', content):  # Statistics
            evidence_count += 2
        if 'peer-reviewed' in content_lower or 'published' in content_lower:
            evidence_count += 2
        
        return min(1.0, evidence_count * 0.1)
    
    def _assess_recency(self, finding: ResearchFinding) -> float:
        """Assess recency of information"""
        # Since we don't have explicit dates, use heuristics
        content = finding.content.lower()
        
        # Look for date indicators
        current_year = datetime.now().year
        recent_indicators = [str(year) for year in range(current_year - 2, current_year + 1)]
        
        if any(indicator in content for indicator in recent_indicators):
            return 0.9
        elif any(str(year) in content for year in range(current_year - 5, current_year - 2)):
            return 0.7
        elif any(str(year) in content for year in range(current_year - 10, current_year - 5)):
            return 0.5
        else:
            return 0.3  # Default for unknown recency
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'relevance': 0.25,
            'accuracy': 0.20,
            'completeness': 0.15,
            'coherence': 0.10,
            'reliability': 0.15,
            'evidence': 0.10,
            'bias': 0.05,  # Bias is inverted (1 - bias_score)
            'recency': 0.05
        }
        
        weighted_score = sum(metrics.get(key, 0.5) * weight for key, weight in weights.items())
        return weighted_score
    
    def _identify_quality_issues(self, metrics: QualityMetrics, finding: ResearchFinding) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        if metrics.relevance_score < 0.6:
            issues.append("Low relevance to research query")
        
        if metrics.accuracy_score < 0.7:
            issues.append("Potential accuracy concerns")
        
        if metrics.source_reliability < 0.6:
            issues.append("Questionable source reliability")
        
        if metrics.bias_score > 0.3:
            issues.append("High bias detected")
        
        if metrics.evidence_strength < 0.5:
            issues.append("Weak supporting evidence")
        
        if metrics.coherence_score < 0.6:
            issues.append("Poor logical coherence")
        
        if metrics.completeness_score < 0.5:
            issues.append("Incomplete information")
        
        if metrics.recency_score < 0.4:
            issues.append("Potentially outdated information")
        
        return issues
    
    def _generate_improvement_suggestions(self, metrics: QualityMetrics, issues: List[str]) -> List[str]:
        """Generate suggestions for quality improvement"""
        suggestions = []
        
        if "Low relevance to research query" in issues:
            suggestions.append("Focus search on more specific keywords related to the research question")
        
        if "Potential accuracy concerns" in issues:
            suggestions.append("Cross-reference findings with multiple authoritative sources")
        
        if "Questionable source reliability" in issues:
            suggestions.append("Prioritize peer-reviewed publications and reputable organizations")
        
        if "High bias detected" in issues:
            suggestions.append("Seek diverse perspectives and counterarguments")
        
        if "Weak supporting evidence" in issues:
            suggestions.append("Look for quantitative data and empirical studies")
        
        if "Poor logical coherence" in issues:
            suggestions.append("Reorganize information to improve logical flow")
        
        if "Incomplete information" in issues:
            suggestions.append("Expand search to cover all aspects of the research question")
        
        if "Potentially outdated information" in issues:
            suggestions.append("Prioritize recent publications and current data")
        
        return suggestions
    
    def _calculate_confidence_interval(self, metrics: QualityMetrics) -> Tuple[float, float]:
        """Calculate confidence interval for quality assessment"""
        # Simple confidence interval based on metric variance
        scores = [
            metrics.relevance_score,
            metrics.accuracy_score,
            metrics.completeness_score,
            metrics.coherence_score,
            metrics.source_reliability,
            metrics.evidence_strength,
            1.0 - metrics.bias_score,  # Invert bias
            metrics.recency_score
        ]
        
        mean_score = mean(scores)
        if len(scores) > 1:
            std_score = stdev(scores)
            margin = 1.96 * std_score / np.sqrt(len(scores))  # 95% confidence
            return (max(0.0, mean_score - margin), min(1.0, mean_score + margin))
        else:
            return (max(0.0, mean_score - 0.1), min(1.0, mean_score + 0.1))
    
    async def evaluate_research_quality(self, research_result: ResearchResult) -> ResearchQualityReport:
        """Comprehensive quality evaluation of research results"""
        print("Conducting comprehensive quality evaluation...")
        
        # Assess individual findings
        finding_assessments = []
        for finding in research_result.findings:
            assessment = await self.assess_finding_quality(finding, research_result.query.original_query)
            finding_assessments.append(assessment)
        
        # Assess source reliability
        unique_sources = list(set(research_result.sources_used))
        source_assessments = []
        
        for source in unique_sources:
            # Find representative content from this source
            source_findings = [f for f in research_result.findings if f.source == source]
            if source_findings:
                sample_content = source_findings[0].content
                reliability = await self._assess_source_reliability(source, sample_content)
                source_assessments.append(reliability)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_research_quality(finding_assessments)
        
        # Generate methodological assessment
        methodological_assessment = self._assess_methodology(research_result, finding_assessments)
        
        # Identify limitations
        limitations = self._identify_research_limitations(research_result, finding_assessments)
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(finding_assessments, source_assessments)
        
        # Assign quality grade
        quality_grade = self._assign_quality_grade(overall_metrics.overall_quality)
        
        # Analyze query quality
        query_analysis = self._analyze_query_quality(research_result.query.original_query)
        
        return ResearchQualityReport(
            query_analysis=query_analysis,
            overall_metrics=overall_metrics,
            finding_assessments=finding_assessments,
            source_reliability=source_assessments,
            methodological_assessment=methodological_assessment,
            limitations_identified=limitations,
            recommendations=recommendations,
            quality_grade=quality_grade
        )
    
    def _calculate_overall_research_quality(self, assessments: List[FindingQuality]) -> QualityMetrics:
        """Calculate overall research quality metrics"""
        if not assessments:
            return QualityMetrics(
                relevance_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                coherence_score=0.0,
                source_reliability=0.0,
                evidence_strength=0.0,
                bias_score=1.0,
                recency_score=0.0,
                overall_quality=0.0
            )
        
        # Calculate weighted averages
        total_weight = sum(assessment.metrics.overall_quality for assessment in assessments)
        
        if total_weight == 0:
            # Unweighted average
            return QualityMetrics(
                relevance_score=mean([a.metrics.relevance_score for a in assessments]),
                accuracy_score=mean([a.metrics.accuracy_score for a in assessments]),
                completeness_score=mean([a.metrics.completeness_score for a in assessments]),
                coherence_score=mean([a.metrics.coherence_score for a in assessments]),
                source_reliability=mean([a.metrics.source_reliability for a in assessments]),
                evidence_strength=mean([a.metrics.evidence_strength for a in assessments]),
                bias_score=mean([a.metrics.bias_score for a in assessments]),
                recency_score=mean([a.metrics.recency_score for a in assessments]),
                overall_quality=mean([a.metrics.overall_quality for a in assessments])
            )
        
        # Weighted average by quality
        return QualityMetrics(
            relevance_score=sum(a.metrics.relevance_score * a.metrics.overall_quality for a in assessments) / total_weight,
            accuracy_score=sum(a.metrics.accuracy_score * a.metrics.overall_quality for a in assessments) / total_weight,
            completeness_score=sum(a.metrics.completeness_score * a.metrics.overall_quality for a in assessments) / total_weight,
            coherence_score=sum(a.metrics.coherence_score * a.metrics.overall_quality for a in assessments) / total_weight,
            source_reliability=sum(a.metrics.source_reliability * a.metrics.overall_quality for a in assessments) / total_weight,
            evidence_strength=sum(a.metrics.evidence_strength * a.metrics.overall_quality for a in assessments) / total_weight,
            bias_score=sum(a.metrics.bias_score * a.metrics.overall_quality for a in assessments) / total_weight,
            recency_score=sum(a.metrics.recency_score * a.metrics.overall_quality for a in assessments) / total_weight,
            overall_quality=mean([a.metrics.overall_quality for a in assessments])
        )
    
    def _assess_methodology(self, research_result: ResearchResult, assessments: List[FindingQuality]) -> str:
        """Assess research methodology quality"""
        issues = []
        strengths = []
        
        # Source diversity
        unique_sources = len(set(research_result.sources_used))
        if unique_sources < 3:
            issues.append("Limited source diversity")
        else:
            strengths.append(f"Good source diversity ({unique_sources} unique sources)")
        
        # Finding quality distribution
        high_quality_findings = sum(1 for a in assessments if a.metrics.overall_quality > 0.7)
        if high_quality_findings / len(assessments) < 0.5:
            issues.append("Low proportion of high-quality findings")
        else:
            strengths.append("Majority of findings meet quality standards")
        
        # Evidence consistency
        evidence_scores = [a.metrics.evidence_strength for a in assessments]
        if stdev(evidence_scores) > 0.3:
            issues.append("Inconsistent evidence quality across findings")
        
        assessment = "Methodology Assessment:\n"
        if strengths:
            assessment += "Strengths: " + "; ".join(strengths) + "\n"
        if issues:
            assessment += "Areas for improvement: " + "; ".join(issues)
        
        return assessment
    
    def _identify_research_limitations(self, research_result: ResearchResult, assessments: List[FindingQuality]) -> List[str]:
        """Identify research limitations"""
        limitations = []
        
        # Sample size limitations
        if len(research_result.findings) < 10:
            limitations.append("Limited number of findings may not provide comprehensive coverage")
        
        # Source limitations
        if len(set(research_result.sources_used)) < 5:
            limitations.append("Limited source diversity may introduce bias")
        
        # Quality limitations
        low_quality_count = sum(1 for a in assessments if a.metrics.overall_quality < 0.6)
        if low_quality_count > len(assessments) * 0.3:
            limitations.append("Significant proportion of findings have quality concerns")
        
        # Recency limitations
        low_recency_count = sum(1 for a in assessments if a.metrics.recency_score < 0.5)
        if low_recency_count > len(assessments) * 0.5:
            limitations.append("Many findings may be outdated")
        
        # Bias limitations
        high_bias_count = sum(1 for a in assessments if a.metrics.bias_score > 0.4)
        if high_bias_count > 0:
            limitations.append("Some findings show signs of bias")
        
        return limitations
    
    def _generate_quality_recommendations(self, finding_assessments: List[FindingQuality], source_assessments: List[SourceReliability]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Based on finding quality
        avg_quality = mean([a.metrics.overall_quality for a in finding_assessments])
        if avg_quality < 0.7:
            recommendations.append("Improve source selection criteria to enhance overall quality")
        
        # Based on source reliability
        avg_reliability = mean([s.reliability_score for s in source_assessments]) if source_assessments else 0.5
        if avg_reliability < 0.7:
            recommendations.append("Prioritize more authoritative and reliable sources")
        
        # Based on common issues
        all_issues = []
        for assessment in finding_assessments:
            all_issues.extend(assessment.quality_issues)
        
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        # Recommend fixes for common issues
        if issue_counts.get("Low relevance to research query", 0) > 2:
            recommendations.append("Refine search strategy to improve relevance of findings")
        
        if issue_counts.get("Potential accuracy concerns", 0) > 2:
            recommendations.append("Implement fact-checking procedures and cross-validation")
        
        if issue_counts.get("High bias detected", 0) > 1:
            recommendations.append("Actively seek diverse perspectives and counterarguments")
        
        return recommendations
    
    def _assign_quality_grade(self, overall_quality: float) -> str:
        """Assign letter grade based on overall quality"""
        if overall_quality >= 0.9:
            return "A"
        elif overall_quality >= 0.8:
            return "B"
        elif overall_quality >= 0.7:
            return "C"
        elif overall_quality >= 0.6:
            return "D"
        else:
            return "F"
    
    def _analyze_query_quality(self, query: str) -> str:
        """Analyze the quality of the original research query"""
        analysis = []
        
        # Length check
        if len(query.split()) < 5:
            analysis.append("Query may be too brief for comprehensive research")
        elif len(query.split()) > 50:
            analysis.append("Query may be too complex; consider breaking into sub-questions")
        else:
            analysis.append("Query length is appropriate for research")
        
        # Specificity check
        vague_words = ['good', 'bad', 'best', 'worst', 'things', 'stuff', 'ways']
        if any(word in query.lower() for word in vague_words):
            analysis.append("Query contains vague terms that may need clarification")
        
        # Question structure
        if '?' in query:
            analysis.append("Well-formed question structure")
        else:
            analysis.append("Consider rephrasing as a specific question")
        
        return "; ".join(analysis)
    
    def export_quality_report(self, quality_report: ResearchQualityReport, output_path: str = None) -> str:
        """Export quality report to JSON"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.REPORTS_DIR / f"quality_report_{timestamp}.json"
        
        # Convert to dictionary
        report_dict = quality_report.model_dump()
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"Quality report exported to: {output_path}")
        return str(output_path)


# Usage example
if __name__ == "__main__":
    async def main():
        from embedding_system import AdvancedEmbeddingSystem
        from research_workflow import ResearchWorkflowOrchestrator
        
        config = ResearchConfig()
        embedding_system = AdvancedEmbeddingSystem(config)
        
        # Add test documents
        result = await embedding_system.add_documents_from_file("realistic_restaurant_reviews.csv")
        print(f"Document loading: {result}")
        
        if result.get("status") == "success":
            # Conduct research
            orchestrator = ResearchWorkflowOrchestrator(embedding_system, config)
            research_result = await orchestrator.orchestrate_research(
                "What factors most influence customer satisfaction in restaurants?"
            )
            
            if research_result['status'] == 'success':
                # Evaluate quality
                quality_controller = QualityController(config)
                quality_report = await quality_controller.evaluate_research_quality(research_result['result'])
                
                print(f"\n=== QUALITY EVALUATION RESULTS ===")
                print(f"Overall Quality Grade: {quality_report.quality_grade}")
                print(f"Overall Quality Score: {quality_report.overall_metrics.overall_quality:.3f}")
                print(f"Relevance: {quality_report.overall_metrics.relevance_score:.3f}")
                print(f"Accuracy: {quality_report.overall_metrics.accuracy_score:.3f}")
                print(f"Source Reliability: {quality_report.overall_metrics.source_reliability:.3f}")
                print(f"Bias Score: {quality_report.overall_metrics.bias_score:.3f}")
                
                print(f"\nLimitations Identified:")
                for limitation in quality_report.limitations_identified:
                    print(f"- {limitation}")
                
                print(f"\nRecommendations:")
                for recommendation in quality_report.recommendations:
                    print(f"- {recommendation}")
                
                # Export report
                report_path = quality_controller.export_quality_report(quality_report)
                print(f"\nFull quality report saved to: {report_path}")
    
    asyncio.run(main())