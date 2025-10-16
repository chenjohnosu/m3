import os
import json
import shutil
import click
import uuid
from datetime import datetime, timezone

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Project-specific components
from core.ingestion.pipeline_factory import get_pipeline
from utils.file_reader import read_files
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager
from pathlib import Path


def get_file_hash(file_path):
    import hashlib
    """Computes the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class VectorManager:
    """
    Manages the vector store and document metadata FOR THE ACTIVE PROJECT.
    """

    def __init__(self, config=None):
        from utils.config import get_config
        self.config = config or get_config()
        self.project_manager = ProjectManager()

        active_project_name, active_project_path = self.project_manager.get_active_project()

        if not active_project_path:
            raise Exception("No active project set. Please use 'm3 project active <name>'.")

        self.project_name = active_project_name
        self.project_path = active_project_path
        self.corpus_path = os.path.join(self.project_path, "corpus")
        self.metadata_path = os.path.join(self.project_path, 'corpus_metadata.json')
        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")

        # Ensure corpus directory exists
        os.makedirs(self.corpus_path, exist_ok=True)

        # --- Configure the embedding model ---
        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml under 'embedding_settings'.")

        click.echo(f"INFO: Loading embedding model '{model_name}'...")
        # Use the global Settings object to configure the embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)

        # --- Configure the LLM ---
        llm_manager = LLMManager(self.config)
        # Use a default LLM from your config (e.g., enrichment_model) for non-pipeline tasks
        Settings.llm = llm_manager.get_llm('enrichment_model')

        # --- Initialize ChromaDB client and LlamaIndex components ---
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection("m3_collection")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)

    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            return {}
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def add_to_corpus(self, paths, doc_type):
        """Copies files to the project's corpus and updates the manifest."""
        metadata = self._load_metadata()
        newly_added_files = []

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                click.secho(f"  > Warning: Path does not exist: {path_str}", fg="yellow")
                continue

            files_to_process = [path] if path.is_file() else list(path.rglob('*'))

            for file_path in files_to_process:
                if not file_path.is_file():
                    continue

                file_hash = get_file_hash(file_path)
                file_id = str(uuid.uuid4())
                destination_path = Path(self.corpus_path) / f"{file_id}{file_path.suffix}"

                # Copy the file to the corpus
                shutil.copy(file_path, destination_path)

                # Update metadata manifest
                metadata[str(destination_path)] = {
                    'original_path': str(file_path),
                    'doc_type': doc_type,
                    'hash': file_hash,
                    'added_at': datetime.now(timezone.utc).isoformat()
                }
                newly_added_files.append(str(destination_path))
                click.echo(f"  > Added '{file_path.name}' to corpus.")

        self._save_metadata(metadata)

        # Now, process only the newly added files for ingestion
        if newly_added_files:
            pipeline = get_pipeline('cogarc', self.config)
            # We need to read the *copied* files for ingestion
            documents = read_files(newly_added_files)
            if not documents:
                click.echo("No valid documents found to ingest.")
                return

            processed_data = pipeline.run(documents, doc_type)

            if 'primary_nodes' in processed_data and processed_data['primary_nodes']:
                self.index.insert_nodes(processed_data['primary_nodes'])
                click.echo(f"  > Stored {len(processed_data['primary_nodes'])} nodes in vector store.")

    def remove_from_corpus(self, filename):
        metadata = self._load_metadata()
        target_path_in_corpus = None

        # Find the file in the manifest by its original name
        for path, meta in metadata.items():
            if Path(meta['original_path']).name == filename:
                target_path_in_corpus = Path(path)
                break

        if not target_path_in_corpus:
            return False, f"File '{filename}' not found in the corpus."

        # LlamaIndex's delete_ref_doc is the proper way to remove nodes
        # It uses the path that was in the document's metadata
        self.index.delete_ref_doc(str(target_path_in_corpus), delete_from_docstore=True)

        # Remove the file from the corpus directory
        if target_path_in_corpus.exists():
            target_path_in_corpus.unlink()

        # Remove the entry from the metadata manifest
        del metadata[str(target_path_in_corpus)]
        self._save_metadata(metadata)

        return True, f"'{filename}' and its associated chunks have been removed from the corpus and vector store."

    def list_corpus(self):
        return self._load_metadata()

    def rebuild_vector_store(self):
        self.create_vector_store(rebuild=True)
        metadata = self._load_metadata()
        if not metadata:
            click.echo("Corpus is empty. Nothing to rebuild.")
            return

        all_files_by_type = {}
        for path, meta in metadata.items():
            doc_type = meta['doc_type']
            # The path to ingest is the key of the metadata dict (the copied file)
            if doc_type not in all_files_by_type:
                all_files_by_type[doc_type] = []
            all_files_by_type[doc_type].append(path)

        for doc_type, paths in all_files_by_type.items():
            click.echo(f"Processing {len(paths)} file(s) of type '{doc_type}'...")
            self.add_to_corpus(paths, doc_type)

        click.secho("✅ Vector store rebuild complete.", fg="green")

    def get_vector_store_status(self):
        """Displays status of the vector store with correct model names and chunk count."""
        click.echo(f"Vector Store Status for Project: '{self.project_name}'")
        click.echo(f"  - Location: {self.chroma_db_path}")

        # --- Corrected Chunk Count Logic ---
        count = self.collection.count()
        click.echo(f"  - Indexed Chunks: {count}")

        # --- Corrected Model Name Logic ---
        ingestion_conf = self.config.get('ingestion_config', {})
        cogarc_settings = ingestion_conf.get('cogarc_settings', {})
        llm_providers = self.config.get('llm_providers', {})

        def get_model_name_from_key(model_key):
            """Looks up the actual model name (e.g., 'llama3') from a role key (e.g., 'synthesis_model')."""
            if not model_key: return "N/A"
            for provider in llm_providers.values():
                if model_key in provider.get('models', {}):
                    return provider['models'][model_key].get('model_name', 'Not Defined')
            return f"'{model_key}' not found in llm_providers"

        synth_model_key = cogarc_settings.get('stage_0_model')
        enrich_model_key = cogarc_settings.get('stage_2_model')

        synth_model_name = get_model_name_from_key(synth_model_key)
        enrich_model_name = get_model_name_from_key(enrich_model_key)

        click.echo("  - Active Ingestion Config:")
        click.echo(f"    - Synthesis Model: {synth_model_name}")
        click.echo(f"    - Enrichment Model: {enrich_model_name}")

    def create_vector_store(self, rebuild=False):
        if os.path.exists(self.chroma_db_path):
            shutil.rmtree(self.chroma_db_path)

        os.makedirs(self.chroma_db_path)
        self.__init__(self.config)  # Re-initialize to create new client/index

        if not rebuild:
            self._save_metadata({})
            click.secho("✅ New blank vector store created.", fg="green")

    def get_file_chunks(self, filename):
        """Retrieves and displays the text chunks for a specific file."""
        metadata = self._load_metadata()
        target_doc_id = None

        # Find the document ID (the path in the corpus) from the manifest
        for doc_id, meta in metadata.items():
            if Path(meta.get('original_path', '')).name == filename:
                target_doc_id = doc_id
                break

        if not target_doc_id:
            click.secho(f"Error: File '{filename}' not found in the corpus manifest.", fg="red")
            return

        # Retrieve nodes from the vector store using a filter
        retriever = self.index.as_retriever(
            vector_store_kwargs={"where": {"file_path": target_doc_id}}
        )
        nodes = retriever.retrieve(" ")  # Use a blank query to get all nodes for the file

        if not nodes:
            click.echo(f"No chunks found for '{filename}'. Has it been ingested?")
            return

        click.secho(f"\n--- Text Chunks for: {filename} ---", bold=True)
        for i, node in enumerate(nodes):
            click.secho(f"\n[Chunk {i + 1}]", fg="yellow")
            click.echo(node.get_content())
        click.secho("\n--- End of Chunks ---", bold=True)

    def query_vector_store(self, query_text):
        click.echo(f"Querying project '{self.project_name}' for: '{query_text}'")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        click.echo(response)