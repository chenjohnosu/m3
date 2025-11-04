import os
import click
import textwrap
import math

from core.project_manager import ProjectManager
from utils.config import get_config
from core.plugin_manager import PluginManager
from core.llm_manager import LLMManager
from llama_index.core.llms import ChatMessage

# LlamaIndex components for querying
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings as LlamaSettings
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

    def __init__(self, config=None, project_name=None, project_path=None, llm_manager=None, plugin_manager=None):

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

        if not plugin_manager:
            self.plugin_manager = PluginManager()
        else:
            self.plugin_manager = plugin_manager
        # --- END FALLBACK ---


        click.echo(f"AnalyzeManager: Operating on project '{self.project_name}'", err=True)

        embed_config = self.config.get('embedding_settings', {})
        model_name = embed_config.get('model_name')
        if not model_name:
            raise ValueError("Embedding model name not found in config.yaml.")

        self.embed_model = get_embed_model(model_name)
        LlamaSettings.embed_model = self.embed_model

        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")

        # --- THIS IS THE FIX ---
        # The check for os.path.exists(self.chroma_db_path) has been REMOVED.
        # We now trust get_chroma_client and get_or_create_collection
        # to handle the creation of the DB, just as VectorManager does.

        self.client = get_chroma_client(self.chroma_db_path)

        self.collection = self.client.get_or_create_collection(
            name="m3_collection",
            metadata=CHROMA_METADATA
        )
        # --- END FIX ---

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        try:
            llm_key = self.config.get('ingestion_config', {}) \
                .get('cogarc_settings', {}) \
                .get('stage_2_model', 'enrichment_model')
            LlamaSettings.llm = self.llm_manager.get_llm(llm_key)
        except Exception as e:
            click.secho(f"Warning: Could not initialize LLMManager: {e}", fg="yellow")
            click.secho("  > LLM-based plugins (summarize, extract, etc.) will not work.", fg="yellow")

    def get_llm(self, model_key='synthesis_model'):
        """
        Retrieves a designated LLM instance from the LLMManager.
        """
        if not self.llm_manager:
            raise Exception("LLMManager is not initialized. Check config.yaml.")

        llm_model_key = self.config.get('ingestion_config', {}) \
            .get('cogarc_settings', {}) \
            .get(model_key, 'synthesis_model')

        click.echo(f"  > Loading LLM: '{llm_model_key}' for analysis...")
        return self.llm_manager.get_llm(llm_model_key)

    def perform_topk_search(self, query_text, k, show_summary=False):
        """
        Core logic for /analyze topk
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
        """
        if threshold == 1.0:
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
        """
        search_scope = "Content + Themes"
        if include_summary:
            search_scope += " + Summary"

        click.echo(f"==> Task: Finding all chunks with exact text '{query_text}'")
        click.echo(f"==> (Searching fields: {search_scope})")

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
            results_meta = self.collection.get(
                where={"holistic_summary": {"$contains": query_text}},
                include=["documents", "metadatas"]
            )

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

            click.echo("  ", nl=False)
            if not math.isnan(score):
                click.secho(f"Score (Cosine Similarity): {score:.4f}", fg="cyan", nl=False)
                click.echo(" (higher is better, 1.0 = perfect match)")
            else:
                click.secho("Score: N/A (Exact Match)", fg="cyan")

            click.echo("  ", nl=False)
            click.secho(f"Source: ", nl=False)
            click.secho(f"{metadata.get('original_filename', 'Unknown')}", fg="green")

            all_keys = list(metadata.keys())
            keys_to_hide = ['original_filename', 'file_path', 'original_text']

            if not show_summary:
                keys_to_hide.append('holistic_summary')

            keys_to_print = [k for k in all_keys if k not in keys_to_hide]

            for key in sorted(keys_to_print):
                if key in metadata:
                    click.echo("  ", nl=False)
                    title_key = key.replace("_", " ").title()
                    click.secho(f"{title_key}: ", nl=False)
                    click.secho(f"{metadata[key]}", fg="magenta")

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


    def list_plugins(self):
        """
        (DEPRECATED) This is now handled by the 'tools' command
        in cli_analyze.py, which uses the session's plugin_manager.
        """
        click.secho("--- Available Analysis Tools ---", bold=True)
        plugins = self.plugin_manager.get_plugins()

        if not plugins:
            click.secho("No plugins found in the 'plugins' directory.", fg="yellow")
            return

        max_key_len = max(len(key) for key in plugins.keys()) if plugins else 0

        for key, plugin in sorted(plugins.items()):
            click.secho(f"  {key:<{max_key_len}}", fg="cyan", nl=False)
            click.echo(f" : {plugin.description}")

    def run_plugin(self, plugin_name: str, **kwargs):
        """
        Core logic for /analyze run <plugin_name>
        Finds and executes the specified analyzer plugin.
        """
        plugin = self.plugin_manager.get_plugin(plugin_name)

        if not plugin:
            click.secho(f"🔥 Error: Plugin '{plugin_name}' not found.", fg="red")
            available = ", ".join(self.plugin_manager.get_plugins().keys())
            click.echo(f"  > Available plugins: {available}")
            return

        try:
            plugin.analyze(self, **kwargs)
        except Exception as e:
            click.secho(f"🔥 Error during plugin execution: {e}", fg="red")
            raise e # Re-raise for the REPL to catch