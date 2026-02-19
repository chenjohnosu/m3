

# m3: The Local-First Qualitative Analysis Toolkit

## User Guide (Short)

-----

### **[Page 1] Introduction**

  * **What is m3?**
      * A local-first, project-based command-line tool for qualitative data analysis.
      * Designed to move beyond simple search (RAG) to *analysis* of a text corpus.
      * Uses local LLMs and embedding models to help you discover themes, map concepts, and synthesize insights from your documents.
  * **Core Concepts**
      * **Project-Based:** All your work (documents, data, vector store) is isolated in a single project directory.
      * **Corpus:** A collection of your original documents (`.txt`, `.pdf`, `.docx`) managed by `m3`.
      * **Cognitive Architect Pipeline:** The "smart" ingestion process that automatically processes your text. It can stratify interviews, generate themes, and create holistic summaries for each document *before* they even go into the vector store.
      * **Analysis Plugins:** Your tools for discovery. You move from basic search (`/a topk`) to advanced, LLM-powered analysis (`/a run clustering`, `/a run interpret`).

-----

### **[Page 2] 1. Setup & Installation**

  * **1.1. Prerequisites (The "Must-Haves")**

      * **Python 3.9+:** Check with `python3 --version`.
      * **Ollama:** `m3` *requires* a running Ollama server to function.
          * Download and install from [https://ollama.com/](https://ollama.com/).
          * Pull the models defined in your config file. By default, you need `mistral` and `intfloat-multilingual-e5-large`.
          * Run: `ollama pull mistral`
          * Run: `ollama pull intfloat-multilingual-e5-large`
      * Ensure the Ollama server is running (check `http://localhost:11434` in your browser).

  * **1.2. Installing m3**

    1.  **Clone the repository:**
        ```bash
        git clone https://github.com/chenjohnosu/m3.git
        cd m3
        ```
    2.  **Create a virtual environment (Recommended):**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    3.  **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```

  * **1.3. First-Time Run & Configuration**

      * The first time you run `m3`, it will create a configuration file at `~/.monkey3/config.yaml`.
      * You can edit this file to change your LLM models (e.g., from `mistral` to `llama3`) or embedding models.
      * For this guide, we will assume the default settings.

-----

### **[Page 3] 2. Quick Start: Your First Analysis in 5 Minutes**

This workflow will guide you through creating a project, ingesting documents, and running your first high-level analysis to summarize the entire corpus.

  * **Step 1: Start Interactive Mode**

      * `m3` is best used in its interactive "REPL" (Read-Eval-Print Loop). All commands inside this mode start with a `/`.
      * Run: `python m3.py --go`
      * *Pro-tip:* You can use aliases: `/p` for `/project`, `/c` for `/corpus`, and `/a` for `/analyze`.

  * **Step 2: Create and Activate Your Project**

      * This creates a new folder in `~/.monkey3/projects/`.
      * `[m3]> /project create my-first-study`
      * The prompt will automatically change to show your active project.
      * `[m3:my-first-study]>`

  * **Step 3: Add Documents to Your Corpus**

      * `m3` copies your files into its managed corpus. Let's add a folder of interview transcripts.
      * The `--type 'interview'` flag is special\! It tells the ingestion pipeline to use its **Q\&A Stratification** logic, which is much more powerful for interviews than the `'document'` type.
      * `[m3:my-first-study]> /c add "path/to/your/interviews/*.docx" --type interview`
      * *(This step automatically runs the full ingestion pipeline, including generating summaries and themes for each document.)*

  * **Step 4: Check the Status**

      * How many files and text chunks do you have?
      * `[m3:my-first-study]> /vector status`
      * You can also see a list of your files:
      * `[m3:my-first-study]> /corpus list`

  * **Step 5: Run Your First Analysis\!**

      * You've ingested 10 interviews. What are they *all* about, as a whole?
      * The `/a run interpret` command finds the **holistic summary** for *every* document in your corpus and feeds them all to an LLM, asking it to create a "meta-summary" of your entire project.
      * `[m3:my-first-study]> /a run interpret`
      * You will now get a 3-5 paragraph synthesis describing the high-level themes, patterns, and insights from your entire collection.

-----

### **[Page 4-5] 3. Command Reference**

A quick reference for all available commands. Run any command without arguments (e.g., `/project`) to see its help.

  * **`/project` (alias: `/p`)**

      * `create <name>`: Creates a new project.
      * `list`: Lists all projects.
      * `active <name>`: Sets the active project for your session.
      * `remove <name>`: Permanently deletes a project.

  * **`/corpus` (alias: `/c`)**

      * `add <path> [--type <name>]`: Adds files/directories. Use `--type` to specify the ingestion pipeline (e.g., `interview`, `document`, `paper`).
      * `list`: Lists all files in the active corpus.
      * `remove <id_or_name>`: Removes a file and its chunks from the vector store.
      * `ingest`: Re-processes all files in the corpus. Use this if you change your config or update files.
      * `summary <id_or_name>`: Displays the holistic summary for a *single* file.

  * **`/vector` (alias: `/v`)**

      * `status`: Shows chunk count, location, and active models.
      * `chunks <id_or_name> [--pretty]`: Dumps all text chunks for a file. Essential for debugging. Use `--pretty` to see all metadata.

  * **`/analyze` (alias: `/a`)**

      * **Phase 1: Retrieval (Finding Chunks)**
          * `topk <query> [--k <N>]`: Finds the Top-K *most similar* chunks to your query.
          * `search <query> [--threshold <0.N>]`: Finds *all* chunks that meet a similarity score (e.g., `0.7`).
          * `exact <query>`: Finds all chunks with an *exact string match*.
      * **Phase 2: Analysis (Understanding Chunks)**
          * `tools`: Lists all available analysis plugins.
          * `run <plugin> [query] [options...]`: Executes an analysis plugin.

  * **Analysis Plugin Options (`/a run ...`)**

      * `--k <N>`: For `clustering` (number of clusters) or `summarize` (number of chunks to summarize).
      * `--threshold <0.N>`: For LLM plugins, the similarity score chunks must meet to be included.
      * `--options "<list>"`: For `entity` (e.g., `--options "People,Locations"`) or `categorize` (e.g., `--options "Positive,Negative"`).
      * `--save`: For `clustering`. This *writes* the analysis results back to your chunks as metadata.

-----

### **[Page 6-8] 4. Use Scenarios: A Qualitative Research Workflow**

`m3` is designed to support an iterative research process. Here’s how you can map `m3` commands to the phases of qualitative inquiry.

  * **Phase 1: Corpus Exploration (Open Coding)**

      * *Your Goal:* "I just added 30 documents. What's in here? What are the main topics and outliers?"
      * **Workflow:**
        1.  **Get the 30,000-foot view:** Run `/a run interpret`. This gives you an immediate, high-level summary of the entire corpus.
        2.  **Find the main "continents" of data:** Run `/a run clustering --k 7`. This will group all your chunks into 7 distinct clusters and use an LLM to generate an "axial theme" for each one. You now have a data-driven thematic map.
        3.  **Create a visual map:** Run `/a run visualize`. This saves a `knowledge_map.png` in your project folder, showing you a 2D t-SNE plot of every chunk. You can visually identify clusters, "islands" of unique topics, and "gaps" in your data.
        4.  **Find the "weird" stuff:** Run `/a run anomaly`. This uses an IsolationForest algorithm to find the chunks that are *least* similar to everything else—perfect for discovering unique ideas, bad data, or off-topic conversations.

  * **Phase 2: Targeted Inquiry (Axial Coding)**

      * *Your Goal:* "My exploration phase suggested 'student isolation' is a key theme. I want to find all discussions about this, summarize them, and understand the sentiment."
      * **Workflow:**
        1.  **Find all relevant chunks:** Use `/a search` for broad retrieval.
              * `/a search "student feelings of isolation and connection" --threshold 0.75`
        2.  **Get a custom summary on just that topic:** Use `/a run summarize`. This command retrieves all chunks above the threshold, then sends them to an LLM to synthesize a new answer.
              * `/a run summarize "What are students saying about isolation?" --threshold 0.75`
        3.  **Analyze the sentiment of those chunks:**
              * `/a run sentiment "student isolation" --threshold 0.75`
        4.  **Categorize the chunks yourself:** Force the LLM to use *your* categories.
              * `/a run categorize "student isolation" --options "Causes of isolation,Coping mechanisms,Suggestions for improvement"`

  * **Phase 3: Iterative Analysis (Selective Coding)**

      * *Your Goal:* "I've confirmed my 5 core themes. I want to permanently 'tag' all chunks in my database with these themes so I can analyze them later."
      * **Workflow:**
        1.  **Run clustering *and save the results*:** This is the key. The `--save` flag *writes* the generated `axial_theme` and `cluster_id` back into the metadata of each chunk.
              * `/a run clustering --k 5 --save`
        2.  **Verify the new metadata:**
              * `/vector chunks "my_interview.docx" --pretty`
              * You will now see `axial_theme: "Student frustration with access"` and `cluster_id: "cluster_2"` in the metadata for chunks in that cluster.
        3.  **You can now query your *own analysis*:** Because the new themes are embedded (per `config.yaml`), your vector search is now more powerful.
              * `/a topk "Student frustration with access"`
              * This will find all chunks you just tagged with that theme, even if the original text didn't contain those exact words.

-----

### **[Page 9] Appendix A: The `config.yaml` File**

  * Located at `~/.monkey3/config.yaml`.
  * **`llm_providers`**: Define your Ollama models. You can add new ones (e.g., `synthesis_model_large: { model_name: 'mixtral' }`) and assign them in the `ingestion_config`.
  * **`ingestion_config`**:
      * `known_doc_types`: Add your own document types (e.g., `literature_review`, `field_notes`).
      * `cogarc_settings`: Assign different LLMs to different pipeline stages.
  * **`analysis_settings`**:
      * `metadata_keys_to_embed`: **(Most Important)** This "allow list" controls what metadata is *searchable*. By default, it includes `themes`, `hypothetical_question`, and `axial_theme`. This is why you can search for the themes you generate.
      * `metadata_keys_to_hide_display`: An "ignore list" to clean up the output of `/vector chunks`.

### **[Page 10] Appendix B: Troubleshooting & Tips**

  * **Error: `connection refused`**
      * Your Ollama server is not running. Start the Ollama application.
  * **Error: `model not found`**
      * You need to pull the model: `ollama pull <model_name>`.
  * **Analysis is slow:**
      * LLM analysis (`interpret`, `summarize`, `clustering`) depends on your local hardware. `synthesis_model` (like `mistral` or `llama3`) is powerful but can be slow. Consider assigning a faster model (like `llama3:8b`) in your `config.yaml` for some tasks.
  * **My search results are bad:**
      * Your ingestion might not be optimized. Did you use the right `--type`?
      * Use `/vector chunks <filename> --pretty` to inspect the text chunks and their metadata. Are the chunks too big? Is the metadata (like `themes`) relevant?
      * If you change `config.yaml` (e.g., add a new `doc_type` or change embedding models), you *must* re-process your corpus:
          * `/c ingest`