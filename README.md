# **m3 (monkey3)**

**m3** is a local-first document analysis toolkit designed for extracting meaningful insights from collections of texts. It provides a robust, project-based workflow to manage, process, and prepare your documents for semantic analysis.

### **Overview**

m3 simplifies the process of analyzing document collections by creating a self-contained environment for each project. It uses a powerful, configurable pipeline to convert documents from multiple formats into a searchable vector store, laying the foundation for advanced analytical queries.

* **Process documents** in multiple formats (.txt, .md, .pdf, .docx).  
* **Manage document versions** intelligently with a content-aware manifest system.  
* **Organize analyses** in self-contained projects.  
* **Utilize modern, local embedding models** without requiring an external server.

### **Key Features**

* **Project-Based Workflow**: All documents, data, and vector stores are neatly organized into distinct project folders.  
* **Multi-Format Ingestion**: Supports .txt, .md, .pdf, and .docx files.  
* **Intelligent Corpus Management**: Automatically tracks file versions using content hashes. When you add an updated file, m3 detects the change and prepares it for re-ingestion, ensuring your analysis is always based on the latest data.  
* **Flexible Content Staging**: Add documents to your project's corpus individually, by directory, or using wildcard patterns (e.g., "reports/\*.txt").  
* **Configurable Embedding Pipeline**: The data ingestion process is controlled by "embedding profiles" defined in a simple YAML configuration. You can easily switch between different models and settings, such as the default high-performance intfloat/multilingual-e5-large.  
* **Self-Contained Architecture**: m3 uses the sentence-transformers library to download and run embedding models locally, removing the need for external dependencies like Ollama.

### **Architecture**

m3 uses a project-centric architecture stored in your home directory (\~/.monkey3/):

1. **Projects**: Each project has its own folder containing a corpus and a vector\_store.  
2. **Corpus**: The corpus directory contains the original documents (copied under a unique ID) and a corpus\_manifest.json file that tracks each file's version, content hash, and unique ID.  
3. **Vector Store**: Each project has a dedicated **ChromaDB** vector store where the embedded text chunks are stored, ready for semantic search.

### **Requirements**

* Python 3.9+  
* Dependencies listed in requirements.txt

#### **Core Dependencies:**

* click (for the command-line interface)  
* PyYAML (for configuration management)  
* chromadb (vector database)  
* llama-index (data processing framework)  
* sentence-transformers & torch (for running local embedding models)  
* PyPDF2 & python-docx (for file parsing)

### **Installation**

\# Clone the repository  
git clone \[https://github.com/yourusername/m3.git\](https://github.com/yourusername/m3.git)  
cd m3

\# Set up virtual environment (recommended)  
python \-m venv venv  
source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

\# Install dependencies  
pip install \-r requirements.txt

### **Quick Start**

\# 1\. Create a new project  
python m3.py project create my\_first\_project

\# 2\. Set it as the active project for the current session  
python m3.py project active my\_first\_project

\# 3\. Add documents to the project's corpus  
\#    (Supports files, directories, and wildcards)  
python m3.py corpus add "path/to/my\_documents/\*.txt"

\# 4\. List the files in the corpus to confirm they were added  
python m3.py corpus list

\# 5\. Process the documents and store them in the vector store  
\#    (This will download the embedding model on the first run)  
python m3.py corpus ingest

\# 6\. Check the status of your vector store  
python m3.py vector status

\# 7\. Start the interactive mode for a streamlined workflow  
python m3.py \--go

### **License**

MIT License

### **Contributing**

We welcome contributions\! Please see CONTRIBUTING.md for details.