# m3 (monkey3) - Development Guide for Claude

## Project Overview
m3 is a local-first document analysis toolkit for qualitative research. It ingests documents (txt, md, pdf, docx), processes them through an LLM-powered pipeline, stores embeddings in ChromaDB, and provides semantic search + analysis plugins.

## Tech Stack
- **CLI**: Click (Python)
- **Vector DB**: ChromaDB (persistent, ~/.monkey3/projects/)
- **Embeddings**: sentence-transformers (HuggingFace, intfloat/multilingual-e5-large)
- **LLM**: Ollama (local, port 11435)
- **RAG/Indexing**: LlamaIndex
- **Testing**: unittest via m3_diag.py harness
- **Config**: YAML (config.yaml, prompts.yaml)

## Architecture
```
m3.py (CLI entry)
  -> cli/  (Click command groups: project, corpus, vector, analyze)
    -> core/  (Business logic managers)
      -> core/ingestion/  (4-stage Cognitive Architect Pipeline)
      -> plugins/  (Analysis plugins: clustering, interpret, entity, etc.)
    -> utils/  (config, device detection, file reading)
```

## Key Files
- `m3.py` - Main CLI entry point, interactive mode (--go), batch mode (--batch)
- `core/session_manager.py` - M3Session: persistent session for interactive mode
- `core/project_manager.py` - Project CRUD, active project tracking
- `core/vector_manager.py` - Document ingestion, ChromaDB operations
- `core/analyze_manager.py` - Search (topk, threshold, exact), plugin execution
- `core/llm_manager.py` - Ollama LLM client caching by model role
- `core/llm_dialogue.py` - Interactive RAG chat loop
- `core/db_manager.py` - Singleton embedding model + ChromaDB client
- `core/ingestion/cognitive_architect_pipeline.py` - 4-stage pipeline orchestrator
- `config.yaml` - Global config template (copied to ~/.monkey3/ on first run)
- `prompts.yaml` - Prompt templates (placeholder)

## Ingestion Pipeline (Cognitive Architect)
1. **Stage 0** (Stratify) - Interview Q&A splitting via LLM (interviews only)
2. **Stage 1** (Structure) - Thematic analysis, extract 2-4 themes per chunk
3. **Stage 2** (Enrich) - Chunk splitting + hypothetical question generation
4. **Stage 3** (Synthesis) - Holistic summary generation per document

## Running the App
```bash
python m3.py --go                    # Interactive mode
python m3.py project create <name>   # Single command mode
python m3.py --batch commands.txt    # Batch mode
```

## Running Tests
```bash
python m3_diag.py          # Run all 130+ tests
python m3_diag.py -v       # Verbose mode
python m3_diag.py --module config  # Single module
```

## Data Storage
- Projects stored in `~/.monkey3/projects/<name>/`
- Each project: `corpus/`, `chroma_db/`, `corpus_metadata.json`
- Config at `~/.monkey3/config.yaml`

## Design Patterns
- **Singleton**: db_manager.py (embedding model, ChromaDB client)
- **Session caching**: M3Session avoids re-init between interactive commands
- **Plugin system**: Auto-discovery of BaseAnalyzerPlugin subclasses
- **Pipeline factory**: Extensible ingestion pipeline creation
- **LLM role mapping**: stratify_model, synthesis_model, enrichment_model

## Conventions
- CLI commands use Click groups with aliases (/p, /c, /v, /a)
- Errors go to stderr (err=True), user output to stdout
- Metadata keys: themes, hypothetical_question, axial_theme, holistic_summary
- Content hashing (SHA-256) for corpus version tracking
- LLM providers served via Ollama on localhost:11435
