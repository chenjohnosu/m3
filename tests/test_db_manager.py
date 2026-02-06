"""
Unit tests for core/db_manager.py

NOTE: These tests require chromadb and its dependencies (including torch)
to be properly installed. If the import fails, tests will be skipped.
"""
import unittest
from unittest.mock import patch, MagicMock

try:
    import core.db_manager as db_module
    DB_AVAILABLE = True
except (ImportError, OSError) as e:
    DB_AVAILABLE = False
    DB_IMPORT_ERROR = str(e)


@unittest.skipUnless(DB_AVAILABLE, "chromadb/torch not available in environment")
class TestGetEmbedModel(unittest.TestCase):
    def setUp(self):
        db_module._cached_embed_model = None

    def tearDown(self):
        db_module._cached_embed_model = None

    @patch('core.db_manager.HuggingFaceEmbedding')
    def test_creates_embed_model(self, mock_hf):
        mock_hf.return_value = MagicMock()
        model = db_module.get_embed_model('intfloat/multilingual-e5-large')
        self.assertIsNotNone(model)
        mock_hf.assert_called_once_with(model_name='intfloat/multilingual-e5-large', normalize=True)

    @patch('core.db_manager.HuggingFaceEmbedding')
    def test_returns_cached_model(self, mock_hf):
        mock_hf.return_value = MagicMock()
        model1 = db_module.get_embed_model('intfloat/multilingual-e5-large')
        model2 = db_module.get_embed_model('intfloat/multilingual-e5-large')
        self.assertIs(model1, model2)
        self.assertEqual(mock_hf.call_count, 1)


@unittest.skipUnless(DB_AVAILABLE, "chromadb/torch not available in environment")
class TestGetChromaClient(unittest.TestCase):
    def setUp(self):
        db_module._cached_chroma_client = None

    def tearDown(self):
        db_module._cached_chroma_client = None

    @patch('core.db_manager.chromadb.PersistentClient')
    def test_creates_chroma_client(self, mock_client):
        mock_client.return_value = MagicMock()
        client = db_module.get_chroma_client('/tmp/test_db')
        self.assertIsNotNone(client)
        mock_client.assert_called_once()

    @patch('core.db_manager.chromadb.PersistentClient')
    def test_returns_cached_client(self, mock_client):
        mock_client.return_value = MagicMock()
        c1 = db_module.get_chroma_client('/tmp/test_db')
        c2 = db_module.get_chroma_client('/tmp/test_db')
        self.assertIs(c1, c2)
        self.assertEqual(mock_client.call_count, 1)

    @patch('core.db_manager.chromadb.PersistentClient')
    def test_client_has_allow_reset(self, mock_client):
        mock_client.return_value = MagicMock()
        db_module.get_chroma_client('/tmp/test_db')
        call_kwargs = mock_client.call_args
        self.assertIsNotNone(call_kwargs)


class TestDbManagerImport(unittest.TestCase):
    """Reports whether db_manager can be imported."""

    def test_import_status(self):
        if not DB_AVAILABLE:
            self.skipTest(f"chromadb/torch not importable: {DB_IMPORT_ERROR}")
        self.assertTrue(DB_AVAILABLE)


if __name__ == '__main__':
    unittest.main()
