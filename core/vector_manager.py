import os
import json
import shutil
import click
import uuid
from datetime import datetime, timezone

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# Project-specific components
from utils.file_reader import read_files
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager
from pathlib import Path

# --- CACHE FOR LAZY LOADING ---
_cached_embed_model = None


def get_file_hash(file_path):
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class VectorManager:
    def __init__(self, config=None):
        global _cached_embed_model
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

        os.makedirs(self.corpus_path, exist_ok=True)

        if _cached_embed_model is None:
            embed_config = self.config.get('embedding_settings', {})
            model_name = embed_config.get('model_name')
            if not model_name:
                raise ValueError("Embedding model name not found in config.yaml.")
            click.echo(f"INFO: Loading embedding model '{model_name}' for the first time...")
            _cached_embed_model = HuggingFaceEmbedding(model_name=model_name)
            Settings.embed_model = _cached_embed_model
        else:
            Settings.embed_model = _cached_embed_model

        llm_manager = LLMManager(self.config)
        Settings.llm = llm_manager.get_llm('enrichment_model')

        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=ChromaSettings(allow_reset=True)
        )
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

    def _find_corpus_file(self, identifier):
        metadata = self._load_metadata()
        for path, meta in metadata.items():
            if Path(path).stem == identifier or Path(meta.get('original_path', '')).name == identifier:
                return path, meta
        return None, None

    def _process_and_ingest_file(self, file_path_in_corpus, doc_type):
        """Processes a single file and ingests it into the vector store."""
        click.echo(f"\n--- Processing '{Path(file_path_in_corpus).name}' (Type: {doc_type}) ---")

        documents = read_files([file_path_in_corpus])
        if not documents:
            click.secho("  > Failed to read document.", fg="yellow")
            return

        if doc_type == 'interview':
            click.echo("  > Running Stage 0: Q&A Stratification (Placeholder)...")

        click.echo("  > Running Stage 2: Chunking...")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        click.echo(f"  > Generated {len(nodes)} text chunks.")

        if nodes:
            self.index.insert_nodes(nodes)
            click.echo(f"  > Stored {len(nodes)} chunks in the vector store.")

        click.echo("--- Finished Processing ---")

    def add_to_corpus(self, paths, doc_type):
        metadata = self._load_metadata()
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
                shutil.copy(file_path, destination_path)
                metadata[str(destination_path)] = {
                    'original_path': str(file_path),
                    'doc_type': doc_type,
                    'hash': file_hash,
                    'added_at': datetime.now(timezone.utc).isoformat()
                }
                click.echo(f"  > Added '{file_path.name}' to corpus manifest.")
                self._save_metadata(metadata)
                self._process_and_ingest_file(str(destination_path), doc_type)

    def remove_from_corpus(self, identifier):
        target_path_in_corpus_str, meta = self._find_corpus_file(identifier)
        if not target_path_in_corpus_str:
            return False, f"File '{identifier}' not found in the corpus."
        target_path_in_corpus = Path(target_path_in_corpus_str)
        original_filename = Path(meta.get('original_path', 'Unknown')).name
        chunk_ids_to_delete = self.collection.get(
            where={"file_path": target_path_in_corpus_str},
            include=[]
        ).get('ids', [])
        if chunk_ids_to_delete:
            self.collection.delete(ids=chunk_ids_to_delete)
        if target_path_in_corpus.exists():
            target_path_in_corpus.unlink()
        metadata = self._load_metadata()
        del metadata[str(target_path_in_corpus)]
        self._save_metadata(metadata)
        return True, f"'{original_filename}' (ID: {target_path_in_corpus.stem}) and its {len(chunk_ids_to_delete)} chunks have been removed."

    def list_corpus(self):
        return self._load_metadata()

    def get_chunk_count(self, doc_id):
        if not doc_id:
            return 0
        result = self.collection.get(where={"file_path": doc_id}, include=[])
        return len(result.get('ids', []))

    def rebuild_vector_store(self):
        click.echo("  > Resetting vector store...")
        self.client.reset()
        click.echo("  > Re-initializing collection and index...")
        self.collection = self.client.get_or_create_collection("m3_collection")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)
        metadata = self._load_metadata()
        if not metadata:
            click.echo("Corpus is empty. Nothing to rebuild.")
            return
        click.echo(f"Found {len(metadata)} files in the corpus to rebuild.")
        for path, meta in metadata.items():
            self._process_and_ingest_file(path, meta.get('doc_type', 'document'))
        click.secho("\n✅ Vector store rebuild complete.", fg="green")

    def get_vector_store_status(self):
        click.echo(f"Vector Store Status for Project: '{self.project_name}'")
        click.echo(f"  - Location: {self.chroma_db_path}")
        count = self.collection.count()
        click.echo(f"  - Indexed Chunks: {count}")
        ingestion_conf = self.config.get('ingestion_config', {})
        cogarc_settings = ingestion_conf.get('cogarc_settings', {})
        llm_providers = self.config.get('llm_providers', {})

        def get_model_name_from_key(model_key):
            if not model_key: return "N/A"
            for provider in llm_providers.values():
                if model_key in provider.get('models', {}):
                    return provider['models'][model_key].get('model_name', 'Not Defined')
            return f"'{model_key}' not found in llm_providers"

        synth_model_name = get_model_name_from_key(cogarc_settings.get('stage_0_model'))
        enrich_model_name = get_model_name_from_key(cogarc_settings.get('stage_2_model'))
        click.echo("  - Active Ingestion Config:")
        click.echo(f"    - Synthesis Model: {synth_model_name}")
        click.echo(f"    - Enrichment Model: {enrich_model_name}")

    def create_vector_store(self, rebuild=False):
        if rebuild:
            click.echo("  > Resetting vector store...")
            self.client.reset()
            self.collection = self.client.get_or_create_collection("m3_collection")
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)
        else:
            if os.path.exists(self.chroma_db_path):
                shutil.rmtree(self.chroma_db_path)
            os.makedirs(self.chroma_db_path)
            self.__init__(self.config)
            self._save_metadata({})
            click.secho("✅ New blank vector store created.", fg="green")

    def get_file_chunks(self, identifier):
        """Retrieves and displays text chunks for a file by its original filename or ID."""
        target_doc_id, meta = self._find_corpus_file(identifier)
        if not target_doc_id:
            click.secho(f"Error: File '{identifier}' not found in the corpus manifest.", fg="red")
            return
        original_filename = Path(meta.get('original_path', 'Unknown')).name

        # --- CORRECTED RETRIEVAL LOGIC ---
        # Directly query ChromaDB for all chunks associated with the file_path.
        # This is the most efficient and scalable method for this diagnostic function.
        results = self.collection.get(
            where={"file_path": target_doc_id},
            include=["documents"]  # Only fetch the document content
        )

        documents = results.get('documents')
        if not documents:
            click.echo(f"No chunks found for '{original_filename}'. Has it been ingested?")
            return

        click.secho(f"\n--- Text Chunks for: {original_filename} (ID: {Path(target_doc_id).stem}) ---", bold=True)
        for i, doc_content in enumerate(documents):
            click.secho(f"\n[Chunk {i + 1}]", fg="yellow")
            click.echo(doc_content)
        click.secho("\n--- End of Chunks ---", bold=True)

    def query_vector_store(self, query_text):
        click.echo(f"Querying project '{self.project_name}' for: '{query_text}'")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        click.echo(response)