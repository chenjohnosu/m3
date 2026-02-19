import os
import json
import shutil
import click
import uuid
import textwrap
from datetime import datetime, timezone
from pathlib import Path
import hashlib

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Project-specific components
from utils.file_reader import read_files
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager
from core.ingestion.pipeline_factory import get_pipeline
from core.db_manager import get_embed_model, get_chroma_client
from utils.config import get_config
from utils.device import detect_device


# E5 models expect cosine similarity
CHROMA_METADATA = {"hnsw:space": "cosine"}
DEFAULT_COLLECTION = "m3_collection"


def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class VectorManager:
    def __init__(self, config=None, project_name=None, project_path=None, llm_manager=None):

        # --- FALLBACK FOR SINGLE-COMMAND/BATCH MODE ---
        if not config:
            self.config = get_config()
        else:
            self.config = config

        if not project_name or not project_path:
            self.project_manager = ProjectManager()
            active_project_name, active_project_path = self.project_manager.get_active_project()
            if not active_project_path:
                raise Exception("No active project set. Please use 'm3 project active <name>'.")
            self.project_name = active_project_name
            self.project_path = active_project_path
        else:
            self.project_name = project_name
            self.project_path = project_path

        if not llm_manager:
            self.llm_manager = LLMManager(self.config)
        else:
            self.llm_manager = llm_manager
        # --- END FALLBACK ---

        self.corpus_path = os.path.join(self.project_path, "corpus")
        self.metadata_path = os.path.join(self.project_path, 'corpus_metadata.json')
        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")
        self.text_cache_path = os.path.join(self.project_path, "text_cache")

        os.makedirs(self.corpus_path, exist_ok=True)
        os.makedirs(self.text_cache_path, exist_ok=True)

        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml.")

        config_device = embed_config.get('device')
        device = detect_device(config_override=config_device if config_device != 'auto' else None)
        self.embed_model = get_embed_model(model_name, device=device)
        LlamaSettings.embed_model = self.embed_model

        # Use the passed-in LLMManager
        LlamaSettings.llm = self.llm_manager.get_llm('enrichment_model')

        analysis_config = self.config.get('analysis_settings', {})
        self.metadata_to_hide = analysis_config.get(
            'metadata_keys_to_hide_display',
            ['original_filename', 'file_path', 'original_text']
        )

        self.client = get_chroma_client(self.chroma_db_path)

        self.collection = self.client.get_or_create_collection(
            name=DEFAULT_COLLECTION,
            metadata=CHROMA_METADATA
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)

    # ── Text Cache Helpers ─────────────────────────────────────────────────

    def _write_text_cache(self, file_id: str, text: str) -> str:
        """Write extracted plain text to text_cache/<file_id>.txt; returns absolute path."""
        cache_file = os.path.join(self.text_cache_path, f"{file_id}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return cache_file

    def _delete_text_cache(self, cache_path):
        """Silently remove a text cache file if it exists."""
        if cache_path:
            Path(cache_path).unlink(missing_ok=True)

    # ── Metadata Helpers ───────────────────────────────────────────────────

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

    # ── Find any entry matching by original_path ───────────────────────────

    def _find_by_original_path(self, source_path_str: str, metadata: dict):
        """Return (corpus_key, meta) for the first entry whose original_path matches source_path_str."""
        for corpus_key, meta in metadata.items():
            if meta.get('original_path') == source_path_str:
                return corpus_key, meta
        return None, None

    # ── Core Ingestion ─────────────────────────────────────────────────────

    def _process_and_ingest_file(self, file_path_in_corpus, doc_type, documents=None):
        """Processes a single file using the Cognitive Architect Pipeline.

        If *documents* is provided (already loaded), skip read_files().
        Tags each node with chunk_index and chunk_count.
        Returns the list of nodes so callers can record chunk_count.
        """
        click.echo(f"\n--- Processing '{Path(file_path_in_corpus).name}' (Type: {doc_type}) ---")

        if documents is None:
            documents = read_files([file_path_in_corpus])
            if not documents:
                click.secho("  > Failed to read document.", fg="yellow")
                return []

        for doc in documents:
            doc.metadata['file_path'] = file_path_in_corpus
            doc.metadata['original_filename'] = Path(file_path_in_corpus).name

        pipeline = get_pipeline('cogarc', self.config, self.llm_manager)

        processed_data = pipeline.run(documents, doc_type)
        nodes = processed_data.get('primary_nodes', [])

        total = len(nodes)
        for i, node in enumerate(nodes):
            node.metadata['chunk_index'] = i
            node.metadata['chunk_count'] = total

        if nodes:
            self.index.insert_nodes(nodes)
            click.echo(f"  > Stored {total} chunks in the vector store.")
        else:
            click.secho("  > No chunks were generated from the document.", fg="yellow")

        click.echo("--- Finished Processing ---")
        return nodes

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
                source_path_str = str(file_path)

                # ── Duplicate / replace detection ──────────────────────────
                existing_key, existing_meta = self._find_by_original_path(source_path_str, metadata)
                inherited_history = []

                if existing_key:
                    if existing_meta.get('hash') == file_hash:
                        click.secho(
                            f"  > Already in corpus (unchanged): '{file_path.name}'. Skipping.",
                            fg="yellow"
                        )
                        continue

                    # Content has changed — prompt user
                    replace = click.confirm(
                        f"  Content has changed for '{file_path.name}'. Replace existing entry?",
                        default=False
                    )

                    if replace:
                        # Build history entry from old metadata
                        history_entry = {
                            'replaced_at': datetime.now(timezone.utc).isoformat(),
                            'old_hash': existing_meta.get('hash'),
                            'old_original_path': existing_meta.get('original_path'),
                            'old_text_cache_path': existing_meta.get('text_cache_path'),
                        }
                        inherited_history = [history_entry] + existing_meta.get('version_history', [])

                        # Delete old ChromaDB chunks, binary, and text cache
                        old_chunk_ids = self.collection.get(
                            where={"file_path": existing_key}, include=[]
                        ).get('ids', [])
                        if old_chunk_ids:
                            self.collection.delete(ids=old_chunk_ids)
                        self._delete_text_cache(existing_meta.get('text_cache_path'))
                        if Path(existing_key).exists():
                            Path(existing_key).unlink()
                        del metadata[existing_key]

                # ── Copy binary and read text ──────────────────────────────
                file_id = str(uuid.uuid4())
                destination_path = Path(self.corpus_path) / f"{file_id}{file_path.suffix}"
                shutil.copy(file_path, destination_path)

                # Read plain text for cache
                documents = read_files([str(destination_path)])
                plain_text = "\n\n".join(doc.get_content() for doc in documents) if documents else ""

                # Write text cache
                cache_path = self._write_text_cache(file_id, plain_text)

                # Build metadata entry with provenance fields
                entry = {
                    'original_path': source_path_str,
                    'doc_type': doc_type,
                    'hash': file_hash,
                    'added_at': datetime.now(timezone.utc).isoformat(),
                    'ingested_at': None,
                    'pipeline': 'cogarc',
                    'chunk_count': 0,
                    'text_cache_path': cache_path,
                }
                if inherited_history:
                    entry['version_history'] = inherited_history

                metadata[str(destination_path)] = entry
                click.echo(f"  > Added '{file_path.name}' to corpus manifest.")
                self._save_metadata(metadata)

                # ── Ingest via pipeline ────────────────────────────────────
                nodes = self._process_and_ingest_file(
                    str(destination_path), doc_type, documents=documents
                )

                # Update provenance fields post-ingestion
                metadata = self._load_metadata()
                if str(destination_path) in metadata:
                    metadata[str(destination_path)]['ingested_at'] = datetime.now(timezone.utc).isoformat()
                    metadata[str(destination_path)]['chunk_count'] = len(nodes)
                    self._save_metadata(metadata)

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
        self._delete_text_cache(meta.get('text_cache_path'))
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

    # ── Provenance / Reconstitution ────────────────────────────────────────

    def get_provenance(self, identifier) -> tuple:
        """Return (True, {corpus_path, **meta}) or (False, error_msg)."""
        corpus_path, meta = self._find_corpus_file(identifier)
        if not corpus_path:
            return False, f"File '{identifier}' not found in the corpus."
        return True, {'corpus_path': corpus_path, **meta}

    def reconstitute_document(self, identifier, from_store=False) -> tuple:
        """Return (True, text) or (False, error_msg).

        Primary: read from text_cache file.
        Fallback (from_store=True or cache missing): reassemble from ChromaDB original_text fields.
        """
        corpus_path, meta = self._find_corpus_file(identifier)
        if not corpus_path:
            return False, f"File '{identifier}' not found in the corpus."

        cache_path = meta.get('text_cache_path')
        if not from_store and cache_path and Path(cache_path).exists():
            with open(cache_path, encoding='utf-8') as f:
                return True, f.read()

        # Fallback: ChromaDB
        results = self.collection.get(
            where={"file_path": corpus_path},
            include=["metadatas"]
        )
        metadatas = results.get('metadatas', [])
        metadatas_sorted = sorted(metadatas, key=lambda m: m.get('chunk_index', 0))
        parts = [m['original_text'] for m in metadatas_sorted if m.get('original_text')]
        if parts:
            return True, '\n\n'.join(parts)
        return False, "No original_text found in vector store chunks."

    def find_source_by_chunk(self, chunk_id: str) -> tuple:
        """Return (True, info_dict) or (False, error_msg)."""
        result = self.collection.get(ids=[chunk_id], include=["metadatas"])
        metadatas = result.get('metadatas', [])
        if not metadatas:
            return False, f"Chunk ID '{chunk_id}' not found in vector store."
        meta = metadatas[0]
        file_path = meta.get('file_path')
        corpus_meta = self._load_metadata().get(file_path, {}) if file_path else {}
        return True, {
            'chunk_id': chunk_id,
            'chunk_index': meta.get('chunk_index'),
            'chunk_count': corpus_meta.get('chunk_count', meta.get('chunk_count')),
            'original_filename': meta.get('original_filename'),
            **corpus_meta,
            'corpus_path': file_path,
        }

    # ── Rebuild / Status ───────────────────────────────────────────────────

    def rebuild_vector_store(self):
        click.echo("  > Resetting vector store...")
        self.client.reset()
        click.echo("  > Re-initializing collection and index...")
        self.collection = self.client.get_or_create_collection(
            name=DEFAULT_COLLECTION,
            metadata=CHROMA_METADATA
        )
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

            keys_to_hide = self.metadata_to_hide[:]

            if not show_summary:
                if 'holistic_summary' not in keys_to_hide:
                    keys_to_hide.append('holistic_summary')
            else:
                if 'holistic_summary' in keys_to_hide:
                    keys_to_hide.remove('holistic_summary')

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
                if should_print_metadata:
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

    # ------------------------------------------------------------------
    # Multi-collection API (for programmatic/facade use)
    # ------------------------------------------------------------------

    def store(self, documents, metadatas, ids, collection_name=DEFAULT_COLLECTION):
        """
        Store documents directly into a named collection, bypassing the pipeline.
        documents: list of strings, metadatas: list of dicts, ids: list of strings.
        """
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata=CHROMA_METADATA
        )
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text, n_results=5, where=None, collection_name=DEFAULT_COLLECTION):
        """
        Semantic query against a named collection. Returns raw Chroma response dict.
        """
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata=CHROMA_METADATA
        )
        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    def delete_by_ids(self, ids, collection_name=DEFAULT_COLLECTION):
        """Delete documents by ID from a named collection."""
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata=CHROMA_METADATA
        )
        collection.delete(ids=ids)
