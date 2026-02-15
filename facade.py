"""
m3/facade.py

Public API for programmatic/import-based use of m3.
All manager internals are accessed through this facade.
CLI and interactive modes are unaffected.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable

# Light imports (always available)
from utils.config import get_config
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager
from core.plugin_manager import PluginManager

# Heavy imports (require chromadb, llama_index, torch) are deferred to methods
# to avoid import failures in environments without full dependencies.

logger = logging.getLogger(__name__)


class M3System:
    """
    High-level facade for m3 document analysis and vector storage.

    Usage:
        from facade import M3System

        m3 = M3System()
        m3.create_project("my_project")
        m3.store("some text", {"key": "value"}, collection_name="entities")
        results = m3.query("what themes relate to identity?", n_results=5)
        m3.close()

    Or as a context manager:
        with M3System() as m3:
            m3.create_project("my_project")
            ...
    """

    def __init__(
        self,
        config: dict | None = None,
        project_name: str | None = None,
    ) -> None:
        """
        Initialize M3System.

        Args:
            config:       Optional config dict to override defaults.
                          If None, loads from ~/.monkey3/config.yaml.
            project_name: If provided, immediately opens this project.
        """
        self._config = config or get_config()
        self._project_manager = ProjectManager()
        self._llm_manager = LLMManager(self._config)
        self._plugin_manager = PluginManager()

        self._current_project: str | None = None
        self._current_project_path: str | None = None
        self._vector_manager = None
        self._analyze_manager = None
        self._pipeline = None

        if project_name:
            self.open_project(project_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_project(self, name: str) -> None:
        """
        Create a new m3 project. Raises if project already exists.
        """
        project_path, message = self._project_manager.init_project(name)
        if project_path is None:
            raise ValueError(message)
        self.open_project(name)
        logger.info(f"Created and opened project '{name}'.")

    def open_project(self, name: str) -> None:
        """
        Open an existing project. Must be called before ingest/query/analyze.
        """
        project_path = self._project_manager.get_project_path_by_name(name)
        if not project_path:
            raise ValueError(f"Project '{name}' not found.")

        from core.vector_manager import VectorManager
        from core.analyze_manager import AnalyzeManager
        from core.ingestion.pipeline_factory import get_pipeline

        self._current_project = name
        self._current_project_path = project_path

        self._vector_manager = VectorManager(
            self._config, name, project_path, self._llm_manager
        )
        self._analyze_manager = AnalyzeManager(
            self._config, name, project_path, self._llm_manager, self._plugin_manager
        )

        # Get a pipeline instance for stage registration
        self._pipeline = get_pipeline('cogarc', self._config, self._llm_manager)

        logger.info(f"Opened project '{name}'.")

    def list_projects(self) -> list[str]:
        """Return names of all existing m3 projects."""
        return self._project_manager.list_projects()

    def delete_project(self, name: str, confirm: bool = False) -> None:
        """
        Delete a project and all its data.
        confirm=True required to prevent accidental deletion.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to delete a project.")
        success, message = self._project_manager.remove_project(name)
        if not success:
            raise ValueError(message)
        if self._current_project == name:
            self._current_project = None
            self._current_project_path = None
            self._vector_manager = None
            self._analyze_manager = None
            self._pipeline = None

    def close(self) -> None:
        """Release resources. Call when done, or use as context manager."""
        self._vector_manager = None
        self._analyze_manager = None
        self._pipeline = None
        self._current_project = None
        self._current_project_path = None

    def __enter__(self) -> "M3System":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Ingestion (full pipeline)
    # ------------------------------------------------------------------

    def ingest(
        self,
        corpus_path: str | Path,
        collection_name: str = "m3_collection",
        doc_type: str = "document",
        batch_mode: bool = True,
    ) -> dict:
        """
        Run the full ingestion pipeline on a corpus directory or file.

        Args:
            corpus_path:       Path to directory of documents or a single file.
            collection_name:   Target Chroma collection.
            doc_type:          Document type for pipeline (e.g., "document", "interview").
            batch_mode:        If True, suppress interactive prompts.

        Returns:
            dict with keys: "documents_processed", "chunks_stored",
                            "collection", "errors"
        """
        self._require_open_project()
        corpus_path = Path(corpus_path).expanduser()
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

        from utils.file_reader import read_files  # lightweight, OK as lazy import

        # Gather files
        if corpus_path.is_file():
            files = [corpus_path]
        else:
            files = [f for f in corpus_path.rglob('*') if f.is_file()]

        errors = []
        total_chunks = 0
        docs_processed = 0

        pipeline = self._pipeline

        for file_path in files:
            try:
                documents = read_files([str(file_path)])
                if not documents:
                    errors.append(f"Failed to read: {file_path}")
                    continue

                for doc in documents:
                    doc.metadata['file_path'] = str(file_path)
                    doc.metadata['original_filename'] = file_path.name

                processed_data = pipeline.run(documents, doc_type)
                nodes = processed_data.get('primary_nodes', [])

                if nodes:
                    # Store into the specified collection
                    doc_texts = [n.get_content() for n in nodes]
                    doc_metas = [n.metadata for n in nodes]
                    doc_ids = [str(uuid.uuid4()) for _ in nodes]
                    self._vector_manager.store(doc_texts, doc_metas, doc_ids, collection_name)
                    total_chunks += len(nodes)

                docs_processed += 1

            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")

        return {
            "documents_processed": docs_processed,
            "chunks_stored": total_chunks,
            "collection": collection_name,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Storage (direct, without full pipeline)
    # ------------------------------------------------------------------

    def store(
        self,
        content: str | list[str],
        metadata: dict | list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "m3_collection",
    ) -> list[str]:
        """
        Store one or more text documents directly into a named collection,
        bypassing the ingestion pipeline.

        Returns:
            List of stored document IDs.
        """
        self._require_open_project()
        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata] * len(content)
        metadata = metadata or [{}] * len(content)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in content]

        self._vector_manager.store(content, metadata, ids, collection_name)
        return ids

    def delete(self, ids: list[str], collection_name: str = "m3_collection") -> None:
        """Delete documents by ID from a named collection."""
        self._require_open_project()
        self._vector_manager.delete_by_ids(ids, collection_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
        collection_name: str = "m3_collection",
    ) -> list[dict]:
        """
        Semantic search over a named collection.

        Returns:
            List of result dicts with: "id", "content", "metadata", "distance"
        """
        self._require_open_project()
        raw = self._vector_manager.query(
            query_text=query_text,
            n_results=n_results,
            where=where,
            collection_name=collection_name,
        )
        return self._normalize_query_results(raw)

    def get_by_id(
        self,
        doc_id: str,
        collection_name: str = "m3_collection",
    ) -> dict | None:
        """Retrieve a single document by ID. Returns None if not found."""
        self._require_open_project()
        from core.db_manager import get_or_create_collection
        collection = get_or_create_collection(self._vector_manager.client, collection_name)
        result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "content": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        plugin_name: str,
        **kwargs,
    ) -> None:
        """
        Run an analysis plugin by name.

        Args:
            plugin_name: Key of the plugin to run (e.g., "clustering", "interpret").
            **kwargs:    Additional arguments passed to the plugin's analyze() method.
        """
        self._require_open_project()
        self._analyze_manager.run_plugin(plugin_name, **kwargs)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return names of all Chroma collections in the current project."""
        self._require_open_project()
        from core.db_manager import list_collections as db_list_collections
        return db_list_collections(self._vector_manager.client)

    def collection_count(self, collection_name: str = "m3_collection") -> int:
        """Return number of documents stored in a named collection."""
        self._require_open_project()
        from core.db_manager import get_or_create_collection
        collection = get_or_create_collection(self._vector_manager.client, collection_name)
        return collection.count()

    def clear_collection(self, collection_name: str, confirm: bool = False) -> None:
        """
        Remove all documents from a collection by deleting and recreating it.
        confirm=True required.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to clear a collection.")
        self._require_open_project()
        from core.db_manager import get_or_create_collection, delete_collection as db_delete_collection
        db_delete_collection(self._vector_manager.client, collection_name)
        get_or_create_collection(self._vector_manager.client, collection_name)

    # ------------------------------------------------------------------
    # Pipeline Extension
    # ------------------------------------------------------------------

    def register_pipeline_stage(
        self,
        name: str,
        stage_fn: Callable,
        position: str = "append",
        description: str = "",
    ) -> None:
        """
        Register an external stage into the ingestion pipeline.
        Must be called after open_project() / create_project().

        Args:
            name:        Unique stage name.
            stage_fn:    Callable(data: dict) -> dict
            position:    "append" | "after:<stage_name>" | "before:<stage_name>"
            description: Human-readable description for logs.
        """
        self._require_open_project()
        self._pipeline.register_stage(
            name=name,
            stage_fn=stage_fn,
            position=position,
            description=description,
        )

    def list_pipeline_stages(self) -> list[dict]:
        """Return ordered list of all pipeline stages (built-in + registered)."""
        self._require_open_project()
        return self._pipeline.list_stages()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open_project(self) -> None:
        if not self._current_project:
            raise RuntimeError(
                "No project is open. Call create_project() or open_project() first."
            )

    @staticmethod
    def _normalize_query_results(raw: dict) -> list[dict]:
        """Normalize raw Chroma query response to list of result dicts."""
        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": distances[i] if i < len(distances) else None,
            })
        return results
