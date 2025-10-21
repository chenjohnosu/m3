import os
import click
import textwrap
import math

from core.project_manager import ProjectManager
from utils.config import get_config
from core.plugin_manager import PluginManager

# --- NEW IMPORTS ---
from core.llm_manager import LLMManager
from llama_index.core.llms import ChatMessage
# --- END NEW IMPORTS ---

# LlamaIndex components for querying
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore, TextNode

# Import the shared instance managers
from core.db_manager import get_embed_model, get_chroma_client

# E5 models expect cosine similarity
CHROMA_METADATA = {"hnsw:space": "cosine"}


class AnalyzeManager:
    """
    Handles all analysis tasks for the currently active project.
    """

    def __init__(self):
        self.project_manager = ProjectManager()
        active_project_name, active_project_path = self.project_manager.get_active_project()

        if not active_project_path:
            raise Exception("No active project set. Please use 'm3 project active <name>'.")

        click.echo(f"AnalyzeManager: Operating on project '{active_project_name}'")
        self.project_name = active_project_name
        self.project_path = active_project_path
        self.config = get_config()

        # --- Initialize the vector store for querying via Global Manager ---
        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml.")

        self.embed_model = get_embed_model(model_name)

        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")
        if not os.path.exists(self.chroma_db_path):
            raise Exception(
                f"Vector store not found for project '{self.project_name}'. Please run '/corpus add' or '/corpus ingest'.")

        self.client = get_chroma_client(self.chroma_db_path)

        self.collection = self.client.get_or_create_collection(
            name="m3_collection",
            metadata=CHROMA_METADATA
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        # --- NEW: Initialize LLM Manager ---
        try:
            self.llm_manager = LLMManager(self.config)
        except Exception as e:
            click.secho(f"Warning: Could not initialize LLMManager: {e}", fg="yellow")
            click.secho("  > LLM-based plugins (summarize, extract, etc.) will not work.", fg="yellow")
            self.llm_manager = None
        # --- END NEW ---

    # --- NEW: Helper method for plugins ---
    def get_llm(self, model_key='synthesis_model'):
        """
        Retrieves a designated LLM instance from the LLMManager.
        'synthesis_model' is used by default for complex analysis.
        """
        if not self.llm_manager:
            raise Exception("LLMManager is not initialized. Check config.yaml.")

        # Look up the model from config (e.g., 'cogarc_settings' -> 'stage_3_model')
        # For simplicity, we'll default to 'synthesis_model'
        llm_model_key = self.config.get('ingestion_config', {}) \
            .get('cogarc_settings', {}) \
            .get('stage_3_model', 'synthesis_model')

        click.echo(f"  > Loading LLM: '{llm_model_key}' for analysis...")
        return self.llm_manager.get_llm(llm_model_key)

    # --- END NEW ---

    def perform_topk_search(self, query_text, k, show_summary=False):
        """
        Core logic for /analyze topk
        Searches the embedded text (Content + Themes by default)
        """
        click.echo(f"==> Task: Finding Top {k} chunks for '{query_text}'")
        click.echo(f"==> (Searching embedded text: Content + Themes)")

        retriever = self.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query_text)

        title = f"Top {len(results)} Results for '{query_text}'"
        self._print_nodes(results, title, show_summary=show_summary)

    def perform_threshold_search(self, query_text, threshold, show_summary=False):
        """
        Core logic for /analyze search
        Searches the embedded text (Content + Themes by default)
        """
        if threshold == 1.0:  # 1.0 was the old default
            threshold = 0.7

        click.echo(f"==> Task: Finding all chunks for '{query_text}' with score > {threshold}")
        click.echo(f"==> (Searching embedded text: Content + Themes)")

        retriever = self.index.as_retriever(similarity_top_k=100)
        all_results = retriever.retrieve(query_text)

        threshold_results = [
            result for result in all_results
            if result.score > threshold
        ]

        title = f"Found {len(threshold_results)} Results for '{query_text}' (Threshold: > {threshold})"
        self._print_nodes(threshold_results, title, show_summary=show_summary)

    def perform_exact_search(self, query_text, include_summary=False):
        """
        Core logic for /analyze exact
        Performs a case-sensitive keyword search.
        If --summary is used, runs two queries and merges results.
        """
        search_scope = "Content + Themes"
        if include_summary:
            search_scope += " + Summary"

        click.echo(f"==> Task: Finding all chunks with exact text '{query_text}'")
        click.echo(f"==> (Searching fields: {search_scope})")

        # Query 1: Search the embedded document text (Content + Themes)
        results_doc = self.collection.get(
            where_document={"$contains": query_text},
            include=["documents", "metadatas"]
        )

        found_ids = set(results_doc.get('ids', []))
        all_results = {
            "ids": results_doc.get('ids', []),
            "documents": results_doc.get('documents', []),
            "metadatas": results_doc.get('metadatas', [])
        }

        if include_summary:
            # Query 2: Search the 'holistic_summary' metadata field
            results_meta = self.collection.get(
                where={"holistic_summary": {"$contains": query_text}},
                include=["documents", "metadatas"]
            )

            # Merge results, avoiding duplicates
            for i, doc_id in enumerate(results_meta.get('ids', [])):
                if doc_id not in found_ids:
                    found_ids.add(doc_id)
                    all_results["ids"].append(doc_id)
                    all_results["documents"].append(results_meta["documents"][i])
                    all_results["metadatas"].append(results_meta["metadatas"][i])

        nodes_with_scores = []
        ids = all_results.get('ids', [])
        documents = all_results.get('documents', [])
        metadatas = all_results.get('metadatas', [])

        for id_val, doc_text, meta in zip(ids, documents, metadatas):
            node = TextNode(id_=id_val, text=doc_text, metadata=meta)
            nodes_with_scores.append(NodeWithScore(node=node, score=float('nan')))

        title = f"Found {len(nodes_with_scores)} Chunks Containing '{query_text}'"
        self._print_nodes(nodes_with_scores, title, show_summary=include_summary)

    def _print_nodes(self, nodes_with_scores: list[NodeWithScore], title: str, show_summary=False):
        """
        Helper function to pretty-print search results.
        Displays 'original_text' from metadata for content.
        Conditionally displays 'holistic_summary'.
        """
        click.secho(f"\n--- {title} ---", bold=True)

        if not nodes_with_scores:
            click.secho("No matching chunks found.", fg="yellow")
            return

        for i, result in enumerate(nodes_with_scores):
            node = result.node
            score = result.score
            metadata = node.metadata

            click.secho(f"\n[Chunk {i + 1}]", fg="yellow", bold=True)

            # --- Print Metadata ---
            click.echo("  ", nl=False)

            if not math.isnan(score):
                click.secho(f"Score (Cosine Similarity): {score:.4f}", fg="cyan", nl=False)
                click.echo(" (higher is better, 1.0 = perfect match)")
            else:
                click.secho("Score: N/A (Exact Match)", fg="cyan")

            click.echo("  ", nl=False)
            click.secho(f"Source: ", nl=False)
            click.secho(f"{metadata.get('original_filename', 'Unknown')}", fg="green")

            # Get ALL metadata keys
            all_keys = list(metadata.keys())

            # Define keys to ALWAYS hide (internal stuff)
            keys_to_hide = ['original_filename', 'file_path', 'original_text']

            # Conditionally HIDE the summary
            if not show_summary:
                keys_to_hide.append('holistic_summary')

            # Filter the list
            keys_to_print = [k for k in all_keys if k not in keys_to_hide]

            for key in sorted(keys_to_print):
                if key in metadata:  # Check if key exists before printing
                    click.echo("  ", nl=False)
                    title = key.replace("_", " ").title()
                    click.secho(f"{title}: ", nl=False)
                    click.secho(f"{metadata[key]}", fg="magenta")

            # --- Print Content ---
            click.secho("  Content:", underline=True)

            content_to_display = metadata.get('original_text', node.get_content())

            content_indent = "    "
            wrapped_content = textwrap.fill(
                content_to_display,
                width=100,
                initial_indent=content_indent,
                subsequent_indent=content_indent
            )
            click.echo(wrapped_content)

        click.secho("\n--- End of Results ---", bold=True)

    # --- PLUGIN METHODS ---

    def list_plugins(self):
        """
        Loads and displays all available analysis plugins.
        """
        click.secho("--- Available Analysis Tools ---", bold=True)
        manager = PluginManager()
        plugins = manager.get_plugins()

        if not plugins:
            click.secho("No plugins found in the 'plugins' directory.", fg="yellow")
            return

        # Find the longest key for formatting
        max_key_len = max(len(key) for key in plugins.keys()) if plugins else 0

        for key, plugin in sorted(plugins.items()):
            click.secho(f"  {key:<{max_key_len}}", fg="cyan", nl=False)
            click.echo(f" : {plugin.description}")

    def run_plugin(self, plugin_name: str, **kwargs):
        """
        Core logic for /analyze run <plugin_name>
        Finds and executes the specified analyzer plugin.
        """
        manager = PluginManager()
        plugin = manager.get_plugin(plugin_name)

        if not plugin:
            click.secho(f"🔥 Error: Plugin '{plugin_name}' not found.", fg="red")
            available = ", ".join(manager.get_plugins().keys())
            click.echo(f"  > Available plugins: {available}")
            return

        try:
            # Pass this instance of AnalyzeManager (self) to the plugin
            # along with any other kwargs (like query_text, k, options)
            plugin.analyze(self, **kwargs)
        except Exception as e:
            click.secho(f"🔥 Error during plugin execution: {e}", fg="red")