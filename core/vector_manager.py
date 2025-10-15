import os
import json
from m3.core.ingestion.pipeline_factory import get_pipeline
from m3.utils.file_reader import read_files
from m3.core.project_manager import ProjectManager


class VectorManager:
    """
    Manages the vector store and document metadata FOR THE ACTIVE PROJECT.
    """

    def __init__(self, config):
        self.config = config
        self.project_manager = ProjectManager()

        active_project_name, active_project_path = self.project_manager.get_active_project()

        if not active_project_path:
            raise Exception("No active project set. Please run 'm3 project init <name>' or 'm3 project use <name>'.")

        print(f"VectorManager operating on active project: '{active_project_name}'")
        self.project_path = active_project_path
        self.metadata_path = os.path.join(self.project_path, 'corpus_metadata.json')

        # --- Initialize your ChromaDB client and Vector Stores here ---
        # The ChromaDB path is now relative to the active project directory
        # self.chroma_db_path = os.path.join(self.project_path, "chroma_db")
        # self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        # self.primary_store = ChromaVectorStore(...)

    def _load_metadata(self):
        """Loads corpus metadata for the active project."""
        if not os.path.exists(self.metadata_path):
            return {}
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        """Saves corpus metadata for the active project."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def add_to_corpus(self, paths, doc_type):
        """Adds documents to the active project's corpus."""
        pipeline = get_pipeline('cogarc', self.config)
        print(f"Reading files to be tagged as '{doc_type}'...")
        documents = read_files(paths)
        if not documents:
            print("No valid documents found at the specified paths.")
            return

        processed_data = pipeline.run(documents, doc_type)

        print("\nStoring processed data in vector store...")
        if 'primary_nodes' in processed_data and processed_data['primary_nodes']:
            # self.primary_index.insert_nodes(processed_data['primary_nodes'])
            print(f"  > Stored {len(processed_data['primary_nodes'])} nodes in the primary collection.")

        metadata = self._load_metadata()
        for doc in documents:
            # Store relative or absolute path as needed
            file_path = doc.metadata.get('file_path', 'Unknown Path')
            metadata[file_path] = {'doc_type': doc_type}
        self._save_metadata(metadata)

        print("Data storage and metadata update complete.")

    def list_corpus(self):
        """Lists all documents in the active project's corpus."""
        return self._load_metadata()