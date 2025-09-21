# Deep Research Agent

A comprehensive research agent with local LLM capabilities, vector database integration, and multi-step reasoning. Built using the LangChain ecosystem, ChromaDB, and Ollama for completely local operation without external APIs.

## 🎯 Problem Statement

**Problem Statement 3: Deep Researcher Agent**

Build a comprehensive research system with:

- **Python-based system** for query handling and response generation
- **Local embedding generation** for document indexing and retrieval
- **Multi-step reasoning** to break down queries into smaller tasks
- **Efficient storage and retrieval** pipeline
- **Enhanced features**: Summarization, interactive refinement, AI-powered explanations, export capabilities

## ✨ Features

### Core Capabilities

- 🔍 **Multi-step reasoning** with query decomposition and synthesis
- 📚 **Multi-format document processing** (PDF, DOCX, CSV, JSON, HTML, Markdown)
- 🧠 **Local embedding generation** using Ollama models (mxbai-embed-large)
- 💾 **Vector storage** with ChromaDB for efficient retrieval
- 🔄 **Hybrid retrieval** combining semantic search, keyword matching, and MMR
- 📊 **Quality assessment** with bias detection and source reliability scoring
- 📄 **Multi-format reports** (Markdown, HTML, PDF) with customizable templates

### Interactive Interfaces

- 🌐 **Streamlit Web Interface** with real-time visualizations
- 💻 **Command-line Interface** for programmatic usage
- 🔧 **REST API** capabilities for integration
- 📱 **Responsive design** with mobile-friendly layouts

### Advanced Features

- 🎯 **Iterative refinement** with quality checkpoints
- 📈 **Research analytics** and confidence scoring
- 🔄 **Follow-up question handling** and context awareness
- 📋 **Research history** and session management
- ⚡ **Async processing** for improved performance

## 🏗️ Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document       │    │  Embedding       │    │  Research       │
│  Processor      │───▶│  System          │───▶│  Agent          │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Quality        │    │  Report          │    │  Interactive    │
│  Controller     │    │  Generator       │    │  Interface      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **AdvancedDocumentProcessor** - Multi-format document loading and preprocessing
2. **AdvancedEmbeddingSystem** - Local embedding generation and vector storage
3. **DeepResearchAgent** - LangGraph-based agent with multi-step reasoning
4. **ResearchWorkflowOrchestrator** - Workflow management with quality assessment
5. **ReportGenerator** - Multi-format report generation with templates
6. **QualityController** - Comprehensive quality evaluation and bias detection
7. **InteractiveInterface** - Web and CLI interfaces for user interaction

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**

   ```bash
   # Download from https://ollama.ai
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

2. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd deep-research-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Basic Usage

#### Command Line Interface

```bash
# Launch interactive CLI
python main.py --interface cli

# Single query research
python main.py --query "What are the key factors for restaurant success?" --depth standard

# Check system status
python main.py --status
```

#### Web Interface

```bash
# Launch Streamlit web interface
python main.py --interface web
# or
streamlit run main.py
```

#### Programmatic Usage

```python
import asyncio
from main import DeepResearchAgent

async def example():
    agent = DeepResearchAgent()

    # Conduct research
    result = await agent.conduct_research(
        "What factors influence customer satisfaction?",
        research_depth="deep"
    )

    print(f"Found {result['summary']['findings_count']} findings")
    print(f"Confidence: {result['summary']['confidence_score']:.3f}")

asyncio.run(example())
```

## 📊 Usage Examples

### 1. Business Analysis

```python
query = "What are the most important factors for restaurant success based on customer feedback?"
result = await agent.conduct_research(query, research_depth="comprehensive")
```

### 2. Academic Research

```python
query = "What does current research show about the effectiveness of machine learning in healthcare?"
result = await agent.conduct_research(query, research_depth="deep")
```

### 3. Market Analysis

```python
query = "Analyze current trends in sustainable technology adoption"
result = await agent.conduct_research(query, research_depth="standard")
```

## 📁 File Structure

```
deep-research-agent/
├── main.py                     # Main application entry point
├── config.py                   # Configuration management
├── document_processor.py       # Document loading and processing
├── embedding_system.py         # Embedding generation and vector storage
├── research_agent.py          # Core research agent with LangGraph
├── advanced_retrieval.py      # Advanced retrieval strategies
├── research_workflow.py       # Workflow orchestration
├── report_generator.py        # Multi-format report generation
├── quality_control.py         # Quality assessment and evaluation
├── interactive_interface.py   # Web and CLI interfaces
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Data directory
│   ├── documents/             # Input documents
│   ├── reports/               # Generated reports
│   └── chroma_db/             # ChromaDB persistence
└── logs/                      # Application logs
```

## ⚙️ Configuration

### Default Configuration

The system uses sensible defaults but can be customized:

```python
# config.py - Key settings
OLLAMA_MODELS = {
    "reasoning": {"model": "llama3.2", "temperature": 0.1},
    "embedding": {"model": "mxbai-embed-large"},
    "summarization": {"model": "llama3.2", "temperature": 0.3}
}

CHROMA_COLLECTION_NAME = "research_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## 🔧 Advanced Features

### Quality Assessment

- **Source reliability scoring** based on domain authority and content quality
- **Bias detection** using linguistic patterns and sentiment analysis
- **Confidence calibration** with statistical confidence intervals
- **Evidence strength evaluation** based on citation patterns and methodology

### Report Generation

- **Multiple templates**: Academic, Business, Executive, Detailed
- **Export formats**: Markdown, HTML, PDF
- **Citation management** with source tracking
- **Custom styling** and branding options

### Interactive Capabilities

- **Follow-up questions** with context awareness
- **Research refinement** based on user feedback
- **Real-time visualizations** of research progress
- **Export and sharing** of research sessions

## 📈 Performance & Scalability

### Optimization Features

- **Async processing** for concurrent operations
- **Batch document processing** for large datasets
- **Intelligent caching** for repeated queries
- **Configurable resource limits** for memory management

### Benchmarks

- **Document processing**: ~100 documents/minute (varies by format)
- **Query processing**: ~2-5 seconds per query (depends on complexity)
- **Memory usage**: ~500MB-2GB (depends on document corpus size)
- **Storage**: ~10-50MB per 1000 documents (compressed embeddings)

## 🛠️ Development

### Project Structure

The codebase follows a modular architecture with clear separation of concerns:

- **Config Layer**: Configuration management and environment setup
- **Processing Layer**: Document loading, embedding generation, quality control
- **Agent Layer**: Research logic, workflow orchestration, reasoning
- **Interface Layer**: User interfaces (web, CLI) and API endpoints
- **Utility Layer**: Logging, error handling, performance monitoring

## 📋 Requirements

### System Requirements

- **Python**: 3.8+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and data
- **Network**: Internet connection for initial model download

### Dependencies

- **LangChain ecosystem**: Core framework and integrations
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM inference
- **Streamlit**: Web interface framework
- **Additional**: See requirements.txt for complete list

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

#### Memory Issues

```python
# Reduce batch size in config
BATCH_SIZE = 5  # Default is 10
CHUNK_SIZE = 500  # Default is 1000
```

#### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --interface cli
```

## 🙏 Acknowledgments

Built with:

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM inference engine
- **Streamlit**: Web interface framework
- **Open Source Community**: Various supporting libraries

---

**Deep Research Agent** - Empowering local, comprehensive research with AI 🔬
