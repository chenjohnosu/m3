import os
import json
import shutil
import click
import uuid
import textwrap
from datetime import datetime, timezone
from pathlib import Path
import hashlib
from typing import List, Tuple

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import VectorStoreQuery

# Project-specific components
from utils.file_reader import read_files
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager, get_embedding_model  # <-- FIX 1: Correct imports
from core.ingestion.pipeline_factory import get_pipeline
from core.db_manager import DBManager  # <-- FIX 2: Import DBManager

# E5 models expect cosine similarity
CHROMA_METADATA = {"hnsw:space": "cosine"}
# Define metadata keys to HIDE by default
METADATA_TO_HIDE_DEFAULT = ['holistic_summary', 'original_filename', 'file_path', 'original_text']


def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class VectorManager:
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

        os.makedirs(self.corpus_path, exist_ok=True)

        # --- FIX 3: Use correct embedding model function ---
        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml.")

        # get_embed_model(model_name) # <-- This was the old, incorrect call
        Settings.embed_model = get_embedding_model(model_name)  # <-- NEW: Set in LlamaIndex Settings

        llm_manager = LLMManager(self.config)
        Settings.llm = llm_manager.get_llm('enrichment_model')

        # --- FIX 4: Initialize Chroma client directly ---
        # self.client = get_chroma_client(self.chroma_db_path) # <-- This function is no longer in db_manager
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)  # <-- NEW: Initialize directly

        self.collection = self.client.get_or_create_collection(
            name="m3_collection",
            metadata=CHROMA_METADATA
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # --- FIX 5: Load index from vector store, don't create new blank one ---
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
        # self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context) # <-- OLD

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
        """Processes a single file using the Cognitive Architect Pipeline."""
        click.echo(f"\n--- Processing '{Path(file_path_in_corpus).name}' (Type: {doc_type}) ---")

        documents = read_files([file_path_in_corpus])
        if not documents:
            click.secho("  > Failed to read document.", fg="yellow")
            return

        # --- FIX 6: Correctly instantiate and pass managers to pipeline ---
        # The new pipeline stages (1 & 2) need the managers, not just the config.
        llm_manager = LLMManager(self.config)
        db_manager = DBManager(self.project_manager)

        # pipeline = get_pipeline('cogarc', self.config) # <-- OLD
        pipeline = get_pipeline('cogarc', llm_manager, db_manager, self, self.project_manager)  # <-- NEW

        processed_data = pipeline.run(documents, doc_type)
        nodes = processed_data.get('primary_nodes', [])

        if nodes:
            # The pipeline's BasePipeline.run() already calls self.add_nodes(nodes)
            # so we don't need self.index.insert_nodes(nodes) here.
            click.echo(f"  > Pipeline processed {len(nodes)} chunks.")
        else:
            click.secho("  > No chunks were generated from the document.", fg="yellow")

        click.echo("--- Finished Processing ---")

    # --- FIX 7: Add methods required by BasePipeline and Stage 2 ---

    def add_nodes(self, nodes: List[BaseNode], doc_id: str):
        """Adds a list of nodes to the vector index. Called by the pipeline."""
        if not nodes:
            return

        click.echo(f"  > Adding {len(nodes)} nodes from {doc_id} to vector store...")
        # insert_nodes is the correct method for the index
        self.index.insert_nodes(nodes)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[BaseNode, float]]:
        """
        Performs a similarity search against the vector store.
        Returns a list of tuples: (node, score)
        Called by Stage 2 Enrich (Paraphrase ID).
        """
        query_embedding = Settings.embed_model.get_query_embedding(query)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k
        )
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        if query_result.nodes:
            for node, score in zip(query_result.nodes, query_result.similarities):
                nodes_with_scores.append((node, score))
        return nodes_with_scores

    def get_all_nodes(self) -> List[BaseNode]:
        """Retrieves all nodes from the vector store."""
        all_ids = self.collection.get(include=[])['ids']
        if not all_ids:
            return []

        results = self.collection.get(ids=all_ids, include=["metadatas", "documents"])
        nodes = []
        for i, doc_id in enumerate(results['ids']):
            # Reconstruct minimal BaseNode objects
            nodes.append(BaseNode(
                id_=doc_id,
                text=results['documents'][i],
                metadata=results['metadatas'][i]
            ))
        return nodes

    # --- End of new methods ---

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
        self.collection = self.client.get_or_create_collection(
            name="m3_collection",
            metadata=CHROMA_METADATA
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )

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
        try:
            collection_metadata = self.collection.metadata
            metric = collection_metadata.get("hnsw:space", "N/A")
            click.echo(f"  - Distance Metric: {metric}")

            count = self.collection.count()
            click.echo(f"  - Indexed Chunks: {count}")
        except Exception as e:
            click.secho(f"🔥 Could not retrieve vector store status: {e}", fg="red")
            return

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
            self.rebuild_vector_store()
        else:
            if os.path.exists(self.chroma_db_path):
                shutil.rmtree(self.chroma_db_path)
            os.makedirs(self.chroma_db_path)
            self.__init__(self.config)
            self._save_metadata({})
            click.secho("✅ New blank vector store created.", fg="green")

    def get_file_chunks(self, identifier, include_metadata=False, pretty=False, show_summary=False):
        target_doc_id, meta = self._find_corpus_file(identifier)
        if not target_doc_id:
            click.secho(f"Error: File '{identifier}' not found in the corpus manifest.", fg="red")
            return
        original_filename = Path(meta.get('original_path', 'Unknown')).name

        results = self.collection.get(
            where={"file_path": target_doc_id},
            include=["documents", "metadatas"]
        )

        items = list(zip(results.get('documents', []), results.get('metadatas', [])))
        if not items:
            click.echo(f"No chunks found for '{original_filename}'. Has it been ingested?")
            return

        click.secho(f"\n--- Text Chunks for: {original_filename} (ID: {Path(target_doc_id).stem}) ---", bold=True)

        for i, (doc_content, doc_meta) in enumerate(items):
            click.secho(f"\n[Chunk {i + 1}]", fg="yellow")

            all_keys = list(doc_meta.keys())
            keys_to_hide = ['original_filename', 'file_path', 'original_text']

            if not show_summary:
                keys_to_hide.append('holistic_summary')

            keys_to_print = [k for k in all_keys if k not in keys_to_hide]
            should_print_metadata = (pretty or include_metadata) and keys_to_print

            if pretty:
                click.secho("  Metadata:", underline=True)
                if should_print_metadata:
                    max_key_len = max(len(key) for key in keys_to_print) if keys_to_print else 0
                    for key in sorted(keys_to_print):
                        value = doc_meta.get(key, "N/A")
                        value_lines = str(value).split('\n')
                        click.echo(f"    - {key:<{max_key_len}} : ", nl=False)
                        click.secho(f"{value_lines[0]}", fg="cyan")
                        for line in value_lines[1:]:
                            click.secho(f"{' ' * (max_key_len + 9)}{line}", fg="cyan")
                else:
                    click.echo("    No metadata to display for this chunk.")

                click.secho("  Content:", underline=True)
                content_to_display = doc_meta.get('original_text', doc_content)
                content_indent = "    "
                wrapped_content = textwrap.fill(
                    content_to_display,
                    width=90,
                    initial_indent=content_indent,
                    subsequent_indent=content_indent
                )
                click.echo(wrapped_content)

            else:  # Not pretty
                if should_print_metadata:  # If --meta flag is present
                    click.secho("  Metadata:", underline=True)
                    for key in sorted(keys_to_print):
                        click.echo(f"    - {key}: ", nl=False)
                        click.secho(f"{doc_meta.get(key, 'N/A')}", fg="cyan")

                click.echo(doc_meta.get('original_text', doc_content))

        click.secho("\n--- End of Chunks ---", bold=True)

    def get_holistic_summary(self, identifier):
        """ Fetches the holistic summary for a single file. """
        target_doc_id, meta = self._find_corpus_file(identifier)
        if not target_doc_id:
            return False, f"File '{identifier}' not found in the corpus manifest."

        original_filename = Path(meta.get('original_path', 'Unknown')).name

        results = self.collection.get(
            where={"file_path": target_doc_id},
            limit=1,
            include=["metadatas"]
        )

        metadatas = results.get('metadatas')
        if not metadatas:
            return False, f"No chunks or metadata found for '{original_filename}'. Has it been ingested?"

        summary = metadatas[0].get('holistic_summary')
        if not summary:
            return False, f"No holistic summary found for '{original_filename}'."

        return True, {"original_name": original_filename, "summary": summary}

    def query_vector_store(self, query_text):
        click.echo(f"Querying project '{self.project_name}' for: '{query_text}'")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        click.echo(response)