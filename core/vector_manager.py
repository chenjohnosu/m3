import click
import shutil
import json
from pathlib import Path
import chromadb
from utils.config import load_config
from utils.file_reader import read_file

# Import LlamaIndex components safely
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.storage.storage_context import StorageContext

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

config = load_config()
M3_BASE_DIR = Path(config.get("base_dir", Path.home() / ".monkey3"))
PROJECTS_DIR = M3_BASE_DIR / "projects"
MANIFEST_NAME = "corpus_manifest.json"


def get_store_path(project_name):
    return PROJECTS_DIR / project_name / "vector_store"


def create_store(project_name):
    store_path = get_store_path(project_name)
    if not (PROJECTS_DIR / project_name).exists():
        click.echo(f"Error: Project '{project_name}' does not exist.", err=True)
        return
    if store_path.exists():
        shutil.rmtree(store_path)
    try:
        store_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(store_path))
        client.create_collection(name=f"{project_name}_collection")
        click.echo(f"  -> Vector store initialized successfully for '{project_name}'.")
    except Exception as e:
        click.echo(f"An unexpected error occurred during store creation: {e}", err=True)


def get_status(project_name):
    store_path = get_store_path(project_name)
    click.echo(f"Checking vector store status for project '{project_name}'...")
    if not store_path.exists() or not any(store_path.iterdir()):
        click.echo("Status: Not initialized.")
        return
    try:
        client = chromadb.PersistentClient(path=str(store_path))
        collection = client.get_collection(name=f"{project_name}_collection")
        click.echo("Status: Initialized")
        click.echo(f"  - Indexed Chunks: {collection.count()}")
    except Exception as e:
        click.echo(f"Error accessing vector store status: {e}", err=True)


def remove_document(project_name, doc_uuid):
    store_path = get_store_path(project_name)
    if not store_path.exists():
        return
    try:
        client = chromadb.PersistentClient(path=str(store_path))
        collection = client.get_collection(name=f"{project_name}_collection")
        collection.delete(where={"uuid": doc_uuid})
        click.echo(f"  -> Removed corresponding entries from vector store.")
    except Exception as e:
        click.echo(f"Warning: Could not remove entries from vector store: {e}", err=True)


def ingest_corpus(project_name):
    """
    Processes files in the corpus and stores them in the vector store.
    """
    if not LLAMA_INDEX_AVAILABLE:
        click.echo(
            "Error: LlamaIndex components not found. Run 'pip install llama-index llama-index-embeddings-ollama llama-index-vector-stores-chroma'.",
            err=True)
        return

    project_path = PROJECTS_DIR / project_name
    corpus_path = project_path / "corpus"
    store_path = get_store_path(project_name)
    manifest_path = corpus_path / MANIFEST_NAME

    if not manifest_path.exists():
        click.echo("Corpus manifest not found. Nothing to ingest.", err=True)
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if not manifest:
        click.echo("Corpus is empty. Add files with '/corpus add' first.")
        return

    click.echo(f"Starting ingestion for project '{project_name}'...")
    try:
        app_config = load_config()
        profile_name = app_config.get('active_profile', 'default_multilingual')
        profile = app_config.get('embedding_profiles', {}).get(profile_name, {})

        Settings.embed_model = OllamaEmbedding(model_name=profile.get('embed_model', 'intfloat/multilingual-e5-large'))
        Settings.node_parser = SentenceSplitter(
            chunk_size=profile.get('chunk_size', 512),
            chunk_overlap=profile.get('chunk_overlap', 50)
        )

        client = chromadb.PersistentClient(path=str(store_path))
        collection = client.get_or_create_collection(name=f"{project_name}_collection")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        ingested_count = 0
        for original_filename, data in manifest.items():
            uuid = data['uuid']
            version = data['version']

            query_result = collection.get(where={"uuid": uuid, "version": version}, limit=1)
            if query_result['ids']:
                click.echo(f"  -> Skipping '{original_filename}' (v{version}), already ingested.")
                continue

            old_versions_result = collection.get(where={"uuid": uuid, "version": {"$lt": version}})
            if old_versions_result['ids']:
                remove_document(project_name, uuid)
                click.echo(f"  -> Found new version of '{original_filename}'. Removing old entries.")

            file_suffix = Path(original_filename).suffix
            internal_filepath = corpus_path / f"{uuid}{file_suffix}"
            _, text_content = read_file(internal_filepath)

            if text_content:
                document = Document(
                    text=text_content,
                    metadata={"filename": original_filename, "uuid": uuid, "version": version}
                )
                VectorStoreIndex.from_documents([document], storage_context=storage_context)
                click.echo(f"  -> Ingested '{original_filename}' (v{version}).")
                ingested_count += 1

        click.echo(f"\nIngestion complete. Processed {ingested_count} new/updated file(s).")

    except Exception as e:
        click.echo(f"An error occurred during ingestion: {e}", err=True)