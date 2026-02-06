"""
Unit tests for core/llm_manager.py
"""
import unittest
from unittest.mock import patch, MagicMock


class TestLLMManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'llm_providers': {
                'ollama_client': {
                    'provider': 'ollama',
                    'base_url': 'http://localhost:11434',
                    'models': {
                        'synthesis_model': {'model_name': 'mistral', 'request_timeout': 120.0},
                        'enrichment_model': {'model_name': 'mistral', 'request_timeout': 60.0},
                        'stratify_model': {'model_name': 'mistral', 'request_timeout': 300.0},
                    }
                }
            },
            'ingestion_config': {
                'cogarc_settings': {
                    'stage_0_model': 'stratify_model',
                    'stage_1_model': 'synthesis_model',
                    'stage_2_model': 'enrichment_model',
                    'stage_3_model': 'synthesis_model',
                }
            }
        }

    @patch('core.llm_manager.Ollama')
    def test_load_clients(self, mock_ollama):
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)
        self.assertIn('ollama_client', manager.clients)
        self.assertEqual(manager.clients['ollama_client']['base_url'], 'http://localhost:11434')

    @patch('core.llm_manager.Ollama')
    def test_get_llm_returns_instance(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)
        llm = manager.get_llm('synthesis_model')
        self.assertIsNotNone(llm)
        mock_ollama.assert_called_once()

    @patch('core.llm_manager.Ollama')
    def test_get_llm_caches_instance(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)
        llm1 = manager.get_llm('synthesis_model')
        llm2 = manager.get_llm('synthesis_model')
        self.assertIs(llm1, llm2)
        # Should only instantiate once
        self.assertEqual(mock_ollama.call_count, 1)

    @patch('core.llm_manager.Ollama')
    def test_get_llm_different_keys(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)
        llm1 = manager.get_llm('synthesis_model')
        llm2 = manager.get_llm('enrichment_model')
        # Different model keys should both work
        self.assertIsNotNone(llm1)
        self.assertIsNotNone(llm2)

    @patch('core.llm_manager.Ollama')
    def test_get_llm_invalid_key_raises(self, mock_ollama):
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)
        with self.assertRaises(ValueError):
            manager.get_llm('nonexistent_model_xyz')

    def test_no_providers_handled(self):
        config = {'llm_providers': {}, 'ingestion_config': {'cogarc_settings': {}}}
        from core.llm_manager import LLMManager
        manager = LLMManager(config)
        self.assertEqual(len(manager.clients), 0)

    @patch('core.llm_manager.Ollama')
    def test_maps_model_roles_correctly(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from core.llm_manager import LLMManager
        manager = LLMManager(self.config)

        # synthesis_model maps to stage_3_model -> synthesis_model
        manager.get_llm('synthesis_model')
        mock_ollama.assert_called_with(
            model='mistral',
            base_url='http://localhost:11434',
            request_timeout=120.0
        )


if __name__ == '__main__':
    unittest.main()
