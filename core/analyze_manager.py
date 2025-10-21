import click
import os
import json
from pathlib import Path

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode
import chromadb  # <-- FIX 1: Import chromadb

# Project-specific components
from core.project_manager import ProjectManager
from core.plugin_manager import PluginManager
from utils.config import get_config
from core.llm_manager import LLMManager, get_embedding_model  # <-- FIX 2: Correct imports

# REMOVED: from core.db_manager import get_embed_model, get_chroma_client

# E5 models expect cosine similarity
CHROMA_METADATA = {"hnsw:space": "cosine"}


class AnalyzeManager:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_manager = ProjectManager()

        active_project_name, active_project_path = self.project_manager.get_active_project()
        if not active_project_path:
            raise Exception("No active project set. Please use 'm3 project active <name>'.")

        self.project_name = active_project_name
        self.project_path = active_project_path
        self.metadata_path = os.path.join(self.project_path, 'corpus_metadata.json')
        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")

        if not os.path.exists(self.chroma_db_path):
            raise Exception(f"Vector store not found for project '{self.project_name}'. Please ingest documents first.")

        # --- FIX 3: Set up embedding model from llm_manager ---
        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml.")

        # Settings.embed_model = get_embed_model(model_name) # <-- Old incorrect call
        Settings.embed_model = get_embedding_model(self.config)  # <-- Corrected call

        # --- FIX 4: Set up default LLM for plugins ---
        llm_manager = LLMManager(self.config)
        # Use a good default model for analysis tasks
        Settings.llm = llm_manager.get_llm('enrichment_model')

        # --- FIX 5: Initialize Chroma client directly ---
        # self.client = get_chroma_client(self.chroma_db_path) # <-- Old incorrect call
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)  # <-- Corrected call

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
        self.query_engine = self.index.as_query_engine()

        self.plugin_manager = PluginManager(self.config, self)

    def list_plugins(self):
        """Lists all available analysis plugins."""
        return self.plugin_manager.list_plugins()

    def run_plugin(self, plugin_name, file_identifier, queries):
        """
        Runs a specific analysis plugin on a given file or the whole corpus.

        :param plugin_name: The name of the plugin to run.
        :param file_identifier: The specific file to analyze, or None for corpus-wide.
        :param queries: A list of query strings for the plugin.
        """
        click.echo(f"  > Running plugin: '{plugin_name}'...")

        target_nodes = self._get_file_nodes(file_identifier)
        if target_nodes is None:
            # Error message already printed by _get_file_nodes
            return

        if not target_nodes:
            target_name = f"file '{file_identifier}'" if file_identifier else "the entire corpus"
            click.secho(f"  > No ingested chunks found for {target_name}. Cannot run plugin.", fg="yellow")
            return

        # Run the plugin
        self.plugin_manager.run_plugin(plugin_name, target_nodes, queries)

    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            return {}
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def _find_corpus_file(self, identifier):
        metadata = self._load_metadata()
        for path, meta in metadata.items():
            if Path(path).stem == identifier or Path(meta.get('original_path', '')).name == identifier:
                return path, meta
        return None, None

    def _get_file_nodes(self, file_identifier) -> list[BaseNode] | None:
        """
        Retrieves all BaseNode objects for a specific file, or all nodes if file_identifier is None.
        Returns None if the file isn't found.
        """
        if file_identifier:
            target_doc_id, meta = self._find_corpus_file(file_identifier)
            if not target_doc_id:
                click.secho(f"Error: File '{file_identifier}' not found in the corpus manifest.", fg="red")
                return None

            click.echo(f"  > Targeting file: '{Path(meta.get('original_path', 'Unknown')).name}'")
            results = self.collection.get(
                where={"file_path": target_doc_id},
                include=["documents", "metadatas"]
            )
        else:
            click.echo("  > Targeting the entire corpus.")
            results = self.collection.get(include=["documents", "metadatas"])

        nodes = []
        if not results.get('ids'):
            return []

        for i, doc_id in enumerate(results['ids']):
            # Reconstruct minimal BaseNode objects for the plugin
            nodes.append(BaseNode(
                id_=doc_id,
                text=results['documents'][i],
                metadata=results['metadatas'][i]
            ))
        return nodes