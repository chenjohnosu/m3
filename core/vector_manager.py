import os
import json
import shutil
import click
from core.ingestion.pipeline_factory import get_pipeline
from utils.file_reader import read_files
from core.project_manager import ProjectManager
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


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
        self.metadata_path = os.path.join(self.project_path, 'corpus_metadata.json')
        self.chroma_db_path = os.path.join(self.project_path, "chroma_db")

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
        pipeline = get_pipeline('cogarc', self.config)
        documents = read_files(paths)
        if not documents:
            print("No valid documents found.")
            return

        processed_data = pipeline.run(documents, doc_type)

        if 'primary_nodes' in processed_data and processed_data['primary_nodes']:
            self.index.insert_nodes(processed_data['primary_nodes'])
            print(f"  > Stored {len(processed_data['primary_nodes'])} nodes.")

        metadata = self._load_metadata()
        for doc in documents:
            file_path = doc.metadata.get('file_path', 'Unknown Path')
            metadata[file_path] = {'doc_type': doc_type}
        self._save_metadata(metadata)

    def remove_from_corpus(self, filename):
        metadata = self._load_metadata()
        target_path = None
        for path in metadata:
            if os.path.basename(path) == filename:
                target_path = path
                break

        if not target_path:
            return False, f"File '{filename}' not found in the corpus."

        # LlamaIndex's delete_ref_doc is the proper way to remove nodes
        self.index.delete_ref_doc(target_path, delete_from_docstore=True)

        del metadata[target_path]
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
        click.echo(f"Retrieving chunks for '{filename}'... (Not yet implemented)")

    def query_vector_store(self, query_text):
        click.echo(f"Querying project '{self.project_name}' for: '{query_text}'")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        click.echo(response)