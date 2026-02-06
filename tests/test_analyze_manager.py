"""
Unit tests for core/analyze_manager.py

NOTE: AnalyzeManager imports chromadb which requires torch.
If torch is broken in the environment, heavy tests will be skipped.
"""
import unittest
import tempfile
import os
import shutil

try:
    from unittest.mock import patch, MagicMock
    from core.analyze_manager import AnalyzeManager
    ANALYZE_IMPORT_OK = True
except (ImportError, OSError) as e:
    ANALYZE_IMPORT_OK = False
    ANALYZE_IMPORT_ERROR = str(e)


@unittest.skipUnless(ANALYZE_IMPORT_OK, "analyze_manager import failed (torch/chromadb)")
class TestAnalyzeManagerInit(unittest.TestCase):
    """Test AnalyzeManager initialization with mocked dependencies."""

    @patch('core.analyze_manager.get_chroma_client')
    @patch('core.analyze_manager.get_embed_model')
    @patch('core.analyze_manager.VectorStoreIndex')
    @patch('core.analyze_manager.ChromaVectorStore')
    @patch('core.analyze_manager.LlamaSettings')
    def test_init_with_valid_config(self, mock_settings, mock_cvs, mock_vsi,
                                     mock_embed, mock_chroma):
        temp_dir = tempfile.mkdtemp()
        try:
            project_path = os.path.join(temp_dir, "test_project")
            os.makedirs(os.path.join(project_path, "chroma_db"), exist_ok=True)

            mock_embed.return_value = MagicMock()
            mock_collection = MagicMock()
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            mock_llm_manager = MagicMock()
            mock_llm_manager.get_llm.return_value = MagicMock()

            mock_plugin_manager = MagicMock()

            config = {
                'embedding_settings': {'model_name': 'test-model'},
                'analysis_settings': {},
                'ingestion_config': {'cogarc_settings': {'stage_2_model': 'enrichment_model'}}
            }

            am = AnalyzeManager(config, "test_project", project_path,
                                mock_llm_manager, mock_plugin_manager)
            self.assertEqual(am.project_name, "test_project")
            self.assertIsNotNone(am.embed_model)
        finally:
            shutil.rmtree(temp_dir)


@unittest.skipUnless(ANALYZE_IMPORT_OK, "analyze_manager import failed (torch/chromadb)")
class TestAnalyzeManagerMethods(unittest.TestCase):
    """Test that expected methods exist on AnalyzeManager."""

    def test_perform_topk_search_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'perform_topk_search'))
        self.assertTrue(callable(getattr(AnalyzeManager, 'perform_topk_search')))

    def test_perform_threshold_search_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'perform_threshold_search'))

    def test_perform_exact_search_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'perform_exact_search'))

    def test_run_plugin_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'run_plugin'))

    def test_get_llm_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'get_llm'))

    def test_list_plugins_exists(self):
        self.assertTrue(hasattr(AnalyzeManager, 'list_plugins'))


class TestAnalyzeManagerImport(unittest.TestCase):
    """Reports whether analyze_manager can be imported."""
    def test_import_status(self):
        if not ANALYZE_IMPORT_OK:
            self.skipTest(f"analyze_manager not importable: {ANALYZE_IMPORT_ERROR}")
        self.assertTrue(ANALYZE_IMPORT_OK)


if __name__ == '__main__':
    unittest.main()
