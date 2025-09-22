# Deep Research Agent 🔬

A comprehensive local research system built with the LangChain ecosystem, ChromaDB vector database, and Ollama for advanced local AI models. This system provides deep research capabilities without relying on external APIs.

## 🌟 Features

- **🔒 100% Local Processing**: No external APIs required
- **📄 Multi-Format Support**: PDF, TXT, DOCX, Markdown files
- **🧠 Multi-Agent Architecture**: Manager, Research, and Tool agents
- **🔍 Deep Reasoning**: Multi-step query decomposition and synthesis
- **💬 Interactive Chat Interface**: Real-time research conversations
- **📊 Export Capabilities**: PDF and Markdown report generation
- **🗂️ Dynamic Vector Management**: Automatic cleanup and session management

## 🏗️ Architecture

### Core Components

1. **Embedding System**: Local embeddings using Ollama's nomic-embed-text
2. **Vector Database**: ChromaDB for persistent document storage
3. **Multi-Agent System**: Coordinated agents for complex research tasks
4. **Reasoning Engine**: IterDRAG-inspired iterative retrieval and analysis
5. **Chat Interface**: Streamlit-based interactive UI
6. **Export System**: Professional report generation

### Agent Architecture

# Deep Research Agent 🔬

A local research system using Ollama models, ChromaDB, and a multi-agent design. This README reflects the current repository layout and entrypoint.

Highlights

- Local-first: runs without external APIs (requires Ollama locally)
- Multi-format document support: PDF, TXT, DOCX, MD, CSV
- Multi-agent architecture: manager, research, and supporting agents
- Streamlit-based UI (entry point: `app.py`)
- Persistent ChromaDB vector store in `data/vector_db`

Repository entry point

- Start the app locally:
  - streamlit run app.py

Primary files and modules

- `app.py` — Streamlit UI entry and session wiring
- `src/chat/chat_interface.py` — Streamlit chat UI and session handling
- `src/agents/` — agent implementations (base, manager, research)
- `src/embeddings/ollama_embeddings.py` — Ollama-based embedding wrapper
- `src/retrieval/vector_store.py` — ChromaDB vector management
- `config/settings.py` — configuration and defaults
- `requirements.txt` — Python dependencies

Quick start

1. Install dependencies:
   - pip install -r requirements.txt
2. Ensure Ollama is running and required models are pulled:
   - ollama pull nomic-embed-text
   - ollama pull llama3.2
   - ollama pull llama3.2:8b
3. Copy and edit env:
   - cp .env.example .env
4. Run the UI:
   - streamlit run app.py

Project structure (relevant)

```
d:/Projects/codemate-om-thakare/
├─ app.py
├─ config/
│  └─ settings.py
├─ src/
│  ├─ agents/
│  │  ├─ base_agent.py
│  │  ├─ manager_agent.py
│  │  └─ research_agent.py
│  ├─ embeddings/
│  │  └─ ollama_embeddings.py
│  ├─ retrieval/
│  │  └─ vector_store.py
│  └─ chat/
│     └─ chat_interface.py
├─ data/
│  └─ vector_db/
└─ requirements.txt
```

Notes & troubleshooting

- If you see startup errors about unexpected kwargs forwarded to base classes (for example `BaseAgent.__init__` receiving agent-specific kwargs), inspect agent constructors in `src/agents/` and ensure only supported args are passed to `super().__init__`. A common fix is to pop agent-specific kwargs before calling the base initializer.
- The UI entry file is `app.py` (not `src/main.py`). Update any scripts or documentation referencing `src/main.py`.
- Config defaults are in `config/settings.py`. Adjust Ollama endpoints, model names, and Chroma persist directory there.

Testing

- If tests exist, run with pytest:
  - pytest

License

- MIT

Contributing

- Fork, add feature branch, include tests, open a PR.

---

This README was updated to match the current code layout and the Streamlit entrypoint at `app.py`.

- Start fresh → New session with clean context
