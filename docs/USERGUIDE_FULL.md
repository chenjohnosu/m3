# m3: The Local-First Qualitative Analysis Toolkit

## Introduction

Welcome to **m3** (monkey3)—a local-first, command-line toolkit designed for qualitative data analysis. m3 helps you manage, process, and analyze collections of text documents (interviews, field notes, articles, etc.) using the power of local Large Language Models (LLMs).

Unlike traditional Q&A tools, m3 supports a true analytical workflow. It moves beyond simple retrieval-augmented generation (RAG) to help you discover latent themes, map conceptual relationships, and synthesize high-level insights across your entire corpus.

### Key Features

- **Local-first design**: All processing happens on your machine—no cloud uploads or external dependencies
- **Intelligent document ingestion**: Automatically extracts themes, generates contextual questions, and creates document summaries
- **Semantic search**: Find relevant content based on meaning, not just keywords
- **Advanced analysis plugins**: Clustering, sentiment analysis, anomaly detection, and more
- **Project-based organization**: Keep multiple studies organized and isolated from one another

---

## Core Concepts

Before diving in, familiarize yourself with these five essential concepts:

**1. Projects**  
m3 is project-based. All files, data, and analysis results for a specific study are stored in a dedicated project folder (by default in `~/.monkey3/projects/`). This keeps your work organized and self-contained.

**2. Corpus**  
Your corpus is the collection of original documents (`.txt`, `.pdf`, `.docx`) you add to a project. m3 copies your files into its managed corpus, leaving your originals untouched.

**3. Vector Store**  
This is your project's "smart" database. When you add documents to the corpus, m3 breaks them into small text chunks and converts those chunks into numerical representations (vectors or "embeddings"). This enables semantic search—finding chunks based on conceptual similarity rather than just keyword matches.

**4. Cognitive Architect Pipeline**  
This is the core innovation in m3. During ingestion, an LLM pre-analyzes your documents and automatically generates enriched metadata including holistic summaries, thematic tags, and hypothetical questions. This preparation happens before you've asked your first question, dramatically improving your analysis.

**5. Analysis Plugins**  
After your data is ingested, you can run various plugins to cluster your data, visualize it, detect anomalies, synthesize answers, and extract entities.

---

## Setup & Installation

m3 runs entirely on your local machine. You'll need three components: Python, the m3 code, and an LLM server (Ollama).

### Step 1: Install and Run Ollama

m3 **requires** a running [Ollama](https://ollama.com/) server to function. Ollama serves the local LLMs and embedding models that m3 uses.

1. **Download and install** Ollama from [ollama.com](https://ollama.com/) for your operating system (macOS, Windows, or Linux).

2. **Launch the Ollama application** and let it run in the background (typically visible in your menu bar or system tray).

3. **Pull the default models** by running these commands in your terminal:

   ```bash
   # Pull the default embedding model
   ollama pull intfloat-multilingual-e5-large
   
   # Pull the default LLM (used for synthesis and analysis)
   ollama pull mistral
   ```

   These models are required for the default m3 setup. You can swap them later by editing `config.yaml`.

### Step 2: Verify Python Version

m3 requires Python 3.9 or newer. Check your version:

```bash
python3 --version
```

If you need to upgrade Python, visit [python.org](https://www.python.org/downloads/).

### Step 3: Install m3

1. **Clone the repository**:

   ```bash
   git clone https://github.com/chenjohnosu/m3.git
   cd m3
   ```

2. **Create a virtual environment** (recommended to avoid dependency conflicts):

   ```bash
   # Create the environment
   python3 -m venv venv
   
   # Activate it (macOS/Linux)
   source venv/bin/activate
   
   # Activate it (Windows)
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Verify Installation

Run m3 for the first time:

```bash
python m3.py
```

This will display the help message and create a default configuration file at `~/.monkey3/config.yaml`. You're now ready to begin!

---

## Quick Start: Your First Analysis in 5 Minutes

This walkthrough will guide you through a complete workflow: creating a project, ingesting documents, and running a high-level analysis.

### Step 1: Launch Interactive Mode

m3 uses an interactive command-line interface called the REPL (Read-Eval-Print Loop). All commands inside this mode start with a forward slash (`/`).

Launch interactive mode:

```bash
python m3.py --go
```

You'll see the prompt: `[m3]>`

**Pro tip**: m3 includes command aliases to save typing:

| Alias | Full Command |
|-------|--------------|
| `/p`  | `/project`   |
| `/c`  | `/corpus`    |
| `/a`  | `/analyze`   |
| `/v`  | `/vector`    |
| `/q`  | `/quit`      |

### Step 2: Create Your First Project

Create a new project called `my-first-study`:

```
[m3]> /project create my-first-study
```

m3 will create the project and set it as active. Your prompt will change to:

```
[m3:my-first-study]>
```

All commands you run now are in the context of this project.

### Step 3: Add Documents (Ingestion)

Prepare a folder containing your documents (interviews, articles, notes, etc.) in `.txt`, `.pdf`, or `.docx` format.

Add your documents using the `/corpus add` command:

```
[m3:my-first-study]> /c add "path/to/your/interviews/*.docx" --type interview
```

The `--type` flag determines which ingestion pipeline to use:

- `--type document`: Use for articles, notes, or general text (default)
- `--type interview`: Use for interview transcripts; activates Q&A extraction

You'll see detailed output as m3 processes your files through the Cognitive Architect Pipeline. This includes extracting themes, generating contextual questions, and creating document summaries.

### Step 4: Check Your Corpus

View the files you've added:

```
[m3:my-first-study]> /corpus list
```

This shows each file's ID, chunk count, type, and original filename.

Check the vector store status:

```
[m3:my-first-study]> /vector status
```

This displays the location, chunk count, and models being used.

### Step 5: Run Your First Analysis

Generate a high-level overview of all your documents:

```
[m3:my-first-study]> /a run interpret
```

m3 will synthesize all document summaries into a "meta-summary" of your entire collection, identifying main themes and patterns.

**Congratulations!** You've completed your first end-to-end analysis with m3.

---

## Understanding the Cognitive Architect Pipeline

The ingestion pipeline is m3's most powerful feature. When you run `/corpus add`, your documents are processed through four intelligent stages:

### Stage 0: Stratify (Q&A Extraction)

**When**: Only runs when you specify `--type interview`

**Action**: The LLM reads the entire document and intelligently extracts distinct question-answer pairs.

**Result**: Instead of one large document chunk, the pipeline works with many smaller, focused "answer" chunks, each tagged with its corresponding question as metadata.

### Stage 1: Structure (Thematic Scaffolding)

**When**: Runs on every chunk

**Action**: The LLM reads the chunk and identifies 2–4 high-level themes.

**Result**: A `themes` metadata field is added (e.g., `"Online learning experiences, Connection to the university, Career preparation"`). This is automated "open coding."

### Stage 2: Enrich (Micro-Context)

**When**: Runs on every chunk from Stage 1

**Action**: The LLM generates a single, concise question that this chunk could answer.

**Result**: A `hypothetical_question` metadata field is added (e.g., `"What challenges do students face with online learning?"`). This dramatically improves search relevance.

### Stage 3: Synthesize (Holistic Summary)

**When**: Runs once per document, using all chunks from that document

**Action**: The LLM synthesizes all chunks into a single abstractive summary.

**Result**: A `holistic_summary` metadata field is added to every chunk belonging to that document.

### Final Step: Embedding

m3 combines the original text with all generated metadata and converts it into a vector using your configured embedding model. By default, m3 embeds:

- The chunk's original text
- Themes (from Stage 1)
- Hypothetical question (from Stage 2)

This combined approach means you can search for literal words, themes, or conceptual questions—dramatically improving search relevance.

---

## Command Reference

### /project (alias: /p)

Manages your analysis projects.

| Command                    | Description                                               |
|----------------------------|-----------------------------------------------------------|
| `/project create <NAME>`   | Create a new project and set it as active                 |
| `/project list`            | List all projects (active project marked with `*`)        |
| `/project active <NAME>`   | Switch to a different project                             |
| `/project remove <NAME>`   | Delete a project and all its data (confirmation required) |
| `/project dialogue <NAME>` | Start an interactive dialogue session (experimental)      |

**Example**:
```
[m3]> /project create my-study
[m3:my-study]> /project list
```

### /corpus (alias: /c)

Manages your document collection.

| Command                            | Description                                     |
|------------------------------------|-------------------------------------------------|
| `/corpus add <PATH> [--type TYPE]` | Add files and run ingestion pipeline            |
| `/corpus list`                     | List all files in the corpus                    |
| `/corpus remove <ID>`              | Remove a file and its chunks                    |
| `/corpus ingest`                   | Re-process all files (use after config changes) |
| `/corpus summary <ID>`             | Display the holistic summary for a file         |

**Flags**:

- `<PATH>`: Single file, directory, or wildcard pattern (e.g., `"data/*.pdf"`)
- `--type <TYPE>`: `interview` or `document` (default)

**Examples**:
```
[m3:my-study]> /c add "interviews/*.docx" --type interview
[m3:my-study]> /corpus list
[m3:my-study]> /corpus summary interview_01.docx
```

### /vector (alias: /v)

Manages your vector store.

| Command                                      | Description                                               |
|----------------------------------------------|-----------------------------------------------------------|
| `/vector status`                             | Show vector store details (location, chunk count, models) |
| `/vector chunks <ID> [--pretty] [--summary]` | Display all chunks for a file                             |
| `/vector query <TEXT>`                       | Run a simple RAG query                                    |

**Flags**:

- `--pretty`: Show all metadata for each chunk
- `--summary`: Include the full holistic summary (hidden by default)

**Examples**:
```
[m3:my-study]> /vector status
[m3:my-study]> /vector chunks interview_01.docx --pretty
[m3:my-study]> /vector query "What do students think about online learning?"
```

### /analyze (alias: /a)

Runs retrieval and analysis on your data. This command has two phases:

#### Phase 1: Retrieval (Finding Chunks)

| Command                                 | Description                                               |
|-----------------------------------------|-----------------------------------------------------------|
| `/a topk "<QUERY>" [--k N]`             | Find top-K most similar chunks (default: 3)               |
| `/a search "<QUERY>" [--threshold 0.N]` | Find all chunks above similarity threshold (default: 0.7) |
| `/a exact "<QUERY>"`                    | Find exact string matches (case-sensitive)                |

#### Phase 2: Analysis (Advanced Plugins)

| Command                             | Description                         |
|-------------------------------------|-------------------------------------|
| `/a tools`                          | List all available analysis plugins |
| `/a run <PLUGIN> [QUERY] [OPTIONS]` | Run a specific plugin               |

**Available plugins**:

| Plugin       | Purpose                                | Example                                                 |
|--------------|----------------------------------------|---------------------------------------------------------|
| `interpret`  | Create a meta-summary of entire corpus | `/a run interpret`                                      |
| `clustering` | Group chunks into k thematic clusters  | `/a run clustering --k 7`                               |
| `visualize`  | Generate 2D knowledge map (t-SNE)      | `/a run visualize`                                      |
| `anomaly`    | Find k outlier chunks                  | `/a run anomaly --k 10`                                 |
| `summarize`  | Create custom summary on a topic       | `/a run summarize "student isolation" --threshold 0.75` |
| `sentiment`  | Analyze sentiment of relevant chunks   | `/a run sentiment "student isolation" --threshold 0.75` |
| `categorize` | Assign chunks to custom categories     | `/a run categorize "topic" --options "Cat1,Cat2,Cat3"`  |
| `entity`     | Extract named entities                 | `/a run entity "topic" --options "People,Locations"`    |

**Examples**:
```
[m3:my-study]> /a topk "What challenges do students face?"
[m3:my-study]> /a search "isolation" --threshold 0.8
[m3:my-study]> /a run clustering --k 5
[m3:my-study]> /a run sentiment "student isolation"
```

---

## Analysis Workflow

m3 is designed to support an iterative research process. Here's how to map commands to different phases of qualitative inquiry.

### Phase 1: Corpus Exploration (Open Coding)

**Goal**: Understand the main topics and patterns in your entire corpus.

**Step 1: Get the big picture**

```
[m3:my-study]> /a run interpret
```

This synthesizes all document summaries into a high-level overview.

**Step 2: Identify main thematic clusters**

```
[m3:my-study]> /a run clustering --k 7
```

This groups chunks into 7 clusters and generates an "axial theme" for each. Adjust `--k` based on your corpus size.

**Step 3: Visualize your data**

```
[m3:my-study]> /a run visualize
```

This generates a 2D knowledge map (`knowledge_map.png`) showing clusters, distances, and outliers.

**Step 4: Find anomalies and outliers**

```
[m3:my-study]> /a run anomaly --k 10
```

This identifies the 10 most unique or unusual chunks—useful for finding unique insights or data quality issues.

### Phase 2: Targeted Inquiry (Axial Coding)

**Goal**: Deep-dive into specific themes and understand them thoroughly.

**Step 1: Create a topic-specific summary**

```
[m3:my-study]> /a run summarize "What are students saying about isolation?" --threshold 0.75
```

This finds all relevant chunks and synthesizes them into a focused summary.

**Step 2: Analyze sentiment**

```
[m3:my-study]> /a run sentiment "student isolation" --threshold 0.75
```

This categorizes relevant chunks as Positive, Negative, Neutral, or Mixed.

**Step 3: Categorize with your codes**

```
[m3:my-study]> /a run categorize "student isolation" --options "Causes,Coping strategies,Solutions"
```

This forces the LLM to assign each chunk to only your specified categories.

**Step 4: Extract key entities**

```
[m3:my-study]> /a run entity "student isolation" --options "People,Organizations,Programs"
```

This performs named entity recognition, extracting only the entity types you need.

### Phase 3: Iterative Analysis (Selective Coding)

**Goal**: Permanently tag chunks with your validated themes to build powerful, layered analysis.

**Step 1: Run clustering with persistence**

```
[m3:my-study]> /a run clustering --k 5 --save
```

The `--save` flag writes analysis results (including `axial_theme` and `cluster_id`) back into the metadata for every chunk.

**Step 2: Verify your new metadata**

```
[m3:my-study]> /vector chunks "my_file.pdf" --pretty
```

You can now see your own analysis (e.g., `axial_theme`) alongside the original text and pipeline-generated themes.

**Step 3: Query your analysis**

```
[m3:my-study]> /a topk "Student frustration with access"
```

Since `axial_theme` is embedded by default, m3 will find all chunks tagged with that theme, even if the original text doesn't contain those exact words.

---

## Configuration & Customization

m3 is controlled by a configuration file at `~/.monkey3/config.yaml`. You can customize models, document types, and analysis behavior.

**Important**: After changing `config.yaml`, run `/corpus ingest` to re-process your data with the new settings.

### Embedding Settings

Control which model creates vectors:

```yaml
embedding_settings:
  provider: 'ollama'
  model_name: 'intfloat-multilingual-e5-large'
```

Available Ollama embedding models include:

- `intfloat-multilingual-e5-large` (default, recommended)
- `nomic-embed-text`
- `mxbai-embed-large`
- `all-minilm`

### LLM Providers

Define available LLMs for analysis:

```yaml
llm_providers:
  ollama_client:
    provider: 'ollama'
    base_url: 'http://localhost:11434'
    models:
      synthesis_model:
        model_name: 'mistral'
        request_timeout: 120.0
      enrichment_model:
        model_name: 'mistral'
        request_timeout: 60.0
      powerful_model:
        model_name: 'llama3:70b'
        request_timeout: 300.0
```

You can add as many models as your hardware supports. Use smaller models (Mistral 7B) for speed; larger models (Llama 70B) for quality.

### Ingestion Configuration

```yaml
known_doc_types:
  - 'document'
  - 'interview'
  - 'paper'
  - 'observation'
  - 'ethnographic_notes'

cogarc_settings:
  stage_0_model: 'synthesis_model'
  stage_1_model: 'synthesis_model'
  stage_2_model: 'enrichment_model'
  stage_3_model: 'powerful_model'    # Use your best model for summaries
```

To add a custom document type, add it to `known_doc_types` and reference it when ingesting with `--type`.

### Analysis Settings

```yaml
analysis_settings:
  metadata_keys_to_embed:
    - 'themes'
    - 'hypothetical_question'
    - 'axial_theme'
  
  metadata_keys_to_hide_display:
    - 'holistic_summary'
```

- **metadata_keys_to_embed**: Only these metadata fields are searchable via semantic search
- **metadata_keys_to_hide_display**: These fields are hidden from query results (use `--summary` flag to reveal)

---

## Troubleshooting

### Error: "Connection refused at http://localhost:11434"

**Cause**: Your Ollama server is not running.

**Solution**: Launch the Ollama application and verify it's running in your menu bar or system tray.

### Error: "model not found"

**Cause**: You haven't pulled the model m3 is trying to use.

**Solution**: Run:
```bash
ollama pull <model_name>
```

For example:
```bash
ollama pull mistral
ollama pull intfloat-multilingual-e5-large
```

Check `config.yaml` for all required model names.

### Analysis is very slow

**Cause**: LLM analysis depends entirely on your local hardware (CPU/GPU/RAM). Heavy tasks like clustering and summarization are computationally demanding.

**Solution**: 

- Be patient (analysis may take 5–60 minutes depending on corpus size and hardware)
- Edit `config.yaml` to use a smaller, faster model for the `synthesis_model` role (e.g., `mistral:7b` instead of `llama3:70b`)
- Run analysis during off-peak hours to free up system resources

### Search results are bad or irrelevant

**Cause 1**: You used the wrong `--type` during ingestion. For example, ingesting an interview with `--type document` will be much less effective.

**Solution 1**: Remove and re-ingest:
```
[m3:my-study]> /corpus remove interview_01.docx
[m3:my-study]> /corpus add "interviews/interview_01.docx" --type interview
```

**Cause 2**: Your `config.yaml` settings are suboptimal (e.g., you changed embedding models).

**Solution 2**: Rebuild the vector store:
```
[m3:my-study]> /corpus ingest
```

### I want to debug what's in my vector store

Use the `/vector chunks` command with `--pretty` to see raw content and metadata:

```
[m3:my-study]> /vector chunks my_file.pdf --pretty --summary
```

This is your primary debugging tool. It shows the text, themes, hypothetical questions, summaries, and any custom metadata you've added. This helps you understand why chunks appear or don't appear in search results.

---

## Tips & Best Practices

1. **Start with `interpret`**: After ingesting your corpus, always run `/a run interpret` first to understand what you're working with.

2. **Use clustering for exploration**: `/a run clustering --k N` is invaluable for discovering unexpected patterns. Start with a conservative k (5–7) and adjust based on results.

3. **Iterate on `--type`**: If you have mixed document types, consider creating custom types in `config.yaml` for better ingestion results.

4. **Save your clustering**: Use `/a run clustering --k N --save` to permanently tag chunks. This enables more powerful searches later.

5. **Use `--threshold` strategically**: Lower thresholds (0.5–0.6) for broad exploration; higher thresholds (0.8–0.9) for precise, focused queries.

6. **Clean your metadata display**: Edit `metadata_keys_to_hide_display` in `config.yaml` to reduce visual clutter in query results.

7. **Monitor hardware**: Keep an eye on CPU/RAM usage during heavy analysis. If performance degrades, consider using smaller models or processing fewer chunks at a time.

8. **Back up your projects**: The `~/.monkey3/projects/` directory contains all your data. Periodically back it up to prevent data loss.

---

## Getting Help

For additional support:

- Check `/project`, `/corpus`, `/analyze`, or any command without arguments to see its help message
- Review the `config.yaml` file for all available options
- Check the [m3 GitHub repository](https://github.com/chenjohnosu/m3) for issues and documentation
- Run `/vector chunks <ID> --pretty` to debug ingestion issues

Happy analyzing!