# core/db_manager.py

import sqlite3
import os
import json
from typing import List, Tuple, Optional, Dict, Any
# This import is no longer needed here, but its removal is optional
# from langchain.schema import Document 
from .project_manager import ProjectManager

class DBManager:
    """
    Manages the SQLite database for a project, storing metadata, chunks,
    and analysis results.
    """

    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager
        self.db_path = os.path.join(self.project_manager.get_project_data_dir(), "m3_project.db")
        self._init_db()

    def _get_connection(self):
        """Establishes a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Document table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_path TEXT,
                doc_type TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Chunk table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_index INTEGER,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
            """)
            
            # Summary table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                summary_text TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
            """)
            
            # NEW: Paraphrase Links table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paraphrase_links (
                chunk_id_a TEXT,
                chunk_id_b TEXT,
                PRIMARY KEY (chunk_id_a, chunk_id_b),
                FOREIGN KEY (chunk_id_a) REFERENCES chunks (chunk_id),
                FOREIGN KEY (chunk_id_b) REFERENCES chunks (chunk_id)
            )
            """)
            
            conn.commit()

    def add_document(self, doc_id: str, file_name: str, file_path: str, doc_type: str):
        """Adds a new document record to the database."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, file_name, file_path, doc_type) VALUES (?, ?, ?, ?)",
                (doc_id, file_name, file_path, doc_type)
            )
            conn.commit()

    def add_chunks(self, doc_id: str, chunks: List): # Type hint changed to List
        """Adds a list of chunks (BaseNodes) for a given document."""
        chunk_data = [
            (
                # Use .node_id for LlamaIndex BaseNode as primary key
                chunk.node_id if chunk.node_id else chunk.metadata['chunk_id'], 
                doc_id,
                chunk.metadata['chunk_index'],
                chunk.get_content(), # Use .get_content()
                json.dumps(chunk.metadata)
            ) for chunk in chunks
        ]
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, content, metadata) VALUES (?, ?, ?, ?, ?)",
                chunk_data
            )
            conn.commit()
    
    def get_document_count(self) -> int:
        """Returns the total number of ingested documents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def get_chunk_count(self) -> int:
        """Returns the total number of chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]

    def get_all_chunks(self) -> List[Dict]: # Return type changed for simplicity
        """Retrieves all chunks from the database as dictionaries."""
        chunks = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT content, metadata FROM chunks ORDER BY doc_id, chunk_index")
            rows = cursor.fetchall()
            for row in rows:
                chunks.append({
                    'content': row['content'],
                    'metadata': json.loads(row['metadata'])
                })
        return chunks

    def add_summary(self, doc_id: str, summary_text: str):
        """Adds a document summary to the database."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO summaries (doc_id, summary_text) VALUES (?, ?)",
                (doc_id, summary_text)
            )
            conn.commit()

    def get_summaries(self, doc_id: str = None) -> List[Tuple[str, str]]:
        """Retrieves summaries, optionally filtered by doc_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute("SELECT doc_id, summary_text FROM summaries WHERE doc_id = ?", (doc_id,))
            else:
                cursor.execute("SELECT doc_id, summary_text FROM summaries")
            return cursor.fetchall()
            
    def document_exists(self, doc_id: str) -> bool:
        """Checks if a document is already in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.fetchone() is not None

    def clear_all_data(self):
        """Deletes all data from all tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM summaries")
            cursor.execute("DELETE FROM paraphrase_links") # Clear new table
            conn.commit()
        
        # Also delete the thematic framework JSON
        framework_path = self._get_framework_path()
        if os.path.exists(framework_path):
            os.remove(framework_path)

    # --- NEW METHODS for Stage 1 & 2 ---

    def add_paraphrase_link(self, chunk_id_a: str, chunk_id_b: str):
        """
        Adds a semantic link (paraphrase) between two chunks.
        Ensures order doesn't matter to prevent duplicates.
        """
        if chunk_id_a > chunk_id_b:
            chunk_id_a, chunk_id_b = chunk_id_b, chunk_id_a
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO paraphrase_links (chunk_id_a, chunk_id_b) VALUES (?, ?)",
                (chunk_id_a, chunk_id_b)
            )
            conn.commit()

    def _get_framework_path(self) -> str:
        """Helper to get the standard file path for the thematic framework."""
        return os.path.join(self.project_manager.get_project_data_dir(), "thematic_framework.json")

    def save_thematic_framework(self, framework: Dict[str, Any]):
        """
        Saves the corpus-wide thematic framework as a JSON file in the project's data directory.
        """
        framework_path = self._get_framework_path()
        try:
            with open(framework_path, 'w', encoding='utf-8') as f:
                json.dump(framework, f, indent=2)
        except IOError as e:
            print(f"Error saving thematic framework: {e}")

    def get_thematic_framework(self) -> Optional[Dict[str, Any]]:
        """
        Loads the corpus-wide thematic framework from its JSON file.
        """
        framework_path = self._get_framework_path()
        if not os.path.exists(framework_path):
            return None
            
        try:
            with open(framework_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading thematic framework: {e}")
            return None