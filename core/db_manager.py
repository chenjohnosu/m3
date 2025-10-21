"""
Manages shared, singleton instances of database clients and models
to prevent "different settings" errors.
"""
import chromadb
import click
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_cached_embed_model = None
_cached_chroma_client = None

def get_embed_model(model_name):
    """
    Returns a cached, singleton instance of the embedding model
    and sets it on LlamaSettings.embed_model.
    """
    global _cached_embed_model
    if _cached_embed_model is None:
        # We use err=True to ensure this logs outside of a --quiet flag
        click.echo(f"INFO: Loading embedding model '{model_name}'...", err=True)
        # E5 models should use normalize=True for cosine similarity
        _cached_embed_model = HuggingFaceEmbedding(model_name=model_name, normalize=True)
        # LlamaSettings.embed_model = _cached_embed_model  <-- REMOVE THIS LINE
    return _cached_embed_model

def get_chroma_client(db_path):
    """
    Returns a cached, singleton instance of the Chroma client.
    The client is always initialized with `allow_reset=True`
    to ensure ingestion and rebuild commands always work.
    """
    global _cached_chroma_client
    if _cached_chroma_client is None:
        click.echo(f"INFO: Initializing ChromaDB PersistentClient at '{db_path}'...", err=True)
        _cached_chroma_client = chromadb.PersistentClient(
            path=db_path,
            # This MUST be True so that VectorManager can reset the DB.
            # It is safe for AnalyzeManager as it never calls reset().
            settings=ChromaSettings(allow_reset=True)
        )
    return _cached_chroma_client