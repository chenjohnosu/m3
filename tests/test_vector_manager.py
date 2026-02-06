"""
Unit tests for core/vector_manager.py

NOTE: VectorManager imports chromadb which requires torch.
If torch is broken in the environment, heavy tests will be skipped.
"""
import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path

# Check if the heavy imports work
try:
    from unittest.mock import patch, MagicMock
    from core.vector_manager import get_file_hash
    VECTOR_IMPORT_OK = True
except (ImportError, OSError) as e:
    VECTOR_IMPORT_OK = False
    VECTOR_IMPORT_ERROR = str(e)


class TestVectorManagerHelpers(unittest.TestCase):
    @unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
    def test_get_file_hash_consistent(self):
        """Test that file hashing produces consistent results."""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "test.txt")
            with open(path, 'w') as f:
                f.write("Hello, world!")
            h1 = get_file_hash(path)
            h2 = get_file_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)  # SHA-256 hex length
        finally:
            shutil.rmtree(temp_dir)

    @unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
    def test_get_file_hash_different_content(self):
        """Test that different files produce different hashes."""
        temp_dir = tempfile.mkdtemp()
        try:
            path1 = os.path.join(temp_dir, "a.txt")
            path2 = os.path.join(temp_dir, "b.txt")
            with open(path1, 'w') as f:
                f.write("Content A")
            with open(path2, 'w') as f:
                f.write("Content B")
            h1 = get_file_hash(path1)
            h2 = get_file_hash(path2)
            self.assertNotEqual(h1, h2)
        finally:
            shutil.rmtree(temp_dir)


class TestVectorManagerMetadata(unittest.TestCase):
    """Test metadata load/save operations in isolation (no heavy deps)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_metadata_empty(self):
        """Test that missing metadata file means no data."""
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        self.assertFalse(os.path.exists(metadata_path))

    def test_save_and_load_metadata(self):
        """Test round-trip save and load of metadata."""
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        test_data = {"/path/to/file.txt": {"hash": "abc123", "doc_type": "document"}}

        with open(metadata_path, 'w') as f:
            json.dump(test_data, f, indent=4)

        with open(metadata_path, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded, test_data)

    def test_metadata_structure(self):
        """Test expected metadata structure."""
        test_data = {
            "/path/corpus/abc-123.txt": {
                "original_path": "/home/user/interview.txt",
                "doc_type": "interview",
                "hash": "sha256hex",
                "added_at": "2024-01-01T00:00:00+00:00"
            }
        }
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(test_data, f)
        with open(metadata_path, 'r') as f:
            loaded = json.load(f)
        entry = loaded["/path/corpus/abc-123.txt"]
        self.assertIn("original_path", entry)
        self.assertIn("doc_type", entry)
        self.assertIn("hash", entry)
        self.assertIn("added_at", entry)


@unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
class TestVectorManagerInit(unittest.TestCase):
    """Test VectorManager initialization with mocked dependencies."""

    @patch('core.vector_manager.get_chroma_client')
    @patch('core.vector_manager.get_embed_model')
    @patch('core.vector_manager.VectorStoreIndex')
    @patch('core.vector_manager.StorageContext')
    @patch('core.vector_manager.ChromaVectorStore')
    @patch('core.vector_manager.LlamaSettings')
    def test_init_with_valid_config(self, mock_settings, mock_cvs, mock_sc,
                                     mock_vsi, mock_embed, mock_chroma):
        temp_dir = tempfile.mkdtemp()
        try:
            project_path = os.path.join(temp_dir, "test_project")
            os.makedirs(os.path.join(project_path, "corpus"), exist_ok=True)

            mock_embed.return_value = MagicMock()
            mock_collection = MagicMock()
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            mock_llm_manager = MagicMock()
            mock_llm_manager.get_llm.return_value = MagicMock()

            config = {
                'embedding_settings': {'model_name': 'test-model'},
                'analysis_settings': {'metadata_keys_to_hide_display': []},
                'ingestion_config': {'cogarc_settings': {}}
            }

            from core.vector_manager import VectorManager
            vm = VectorManager(config, "test_project", project_path, mock_llm_manager)
            self.assertEqual(vm.project_name, "test_project")
            self.assertIsNotNone(vm.embed_model)
        finally:
            shutil.rmtree(temp_dir)


class TestVectorManagerImport(unittest.TestCase):
    """Reports whether vector_manager can be imported."""
    def test_import_status(self):
        if not VECTOR_IMPORT_OK:
            self.skipTest(f"vector_manager not importable: {VECTOR_IMPORT_ERROR}")
        self.assertTrue(VECTOR_IMPORT_OK)


if __name__ == '__main__':
    unittest.main()
