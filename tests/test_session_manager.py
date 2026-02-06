"""
Unit tests for core/session_manager.py

NOTE: SessionManager imports VectorManager/AnalyzeManager which import chromadb.
If torch is broken in the environment, tests will be skipped.
"""
import unittest

try:
    from unittest.mock import patch, MagicMock
    from core.session_manager import M3Session
    SESSION_IMPORT_OK = True
except (ImportError, OSError) as e:
    SESSION_IMPORT_OK = False
    SESSION_IMPORT_ERROR = str(e)


@unittest.skipUnless(SESSION_IMPORT_OK, "session_manager import failed (torch/chromadb)")
class TestM3Session(unittest.TestCase):
    """Test M3Session initialization and project loading."""

    @patch('core.session_manager.AnalyzeManager')
    @patch('core.session_manager.VectorManager')
    @patch('core.session_manager.PluginManager')
    @patch('core.session_manager.LLMManager')
    @patch('core.session_manager.ProjectManager')
    @patch('core.session_manager.get_config')
    def test_session_init_no_active_project(self, mock_config, mock_pm, mock_llm,
                                             mock_plugin, mock_vm, mock_am):
        """Session should initialize cleanly with no active project."""
        mock_config.return_value = {
            'project_settings': {'projects_directory': '/tmp/projects'},
            'llm_providers': {},
            'ingestion_config': {}
        }
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_active_project.return_value = (None, None)
        mock_pm.return_value = mock_pm_instance

        session = M3Session()

        self.assertIsNone(session.active_project_name)
        self.assertIsNone(session.vector_manager)
        self.assertIsNone(session.analyze_manager)

    @patch('core.session_manager.AnalyzeManager')
    @patch('core.session_manager.VectorManager')
    @patch('core.session_manager.PluginManager')
    @patch('core.session_manager.LLMManager')
    @patch('core.session_manager.ProjectManager')
    @patch('core.session_manager.get_config')
    def test_session_init_with_active_project(self, mock_config, mock_pm, mock_llm,
                                               mock_plugin, mock_vm, mock_am):
        """Session should load the active project on startup."""
        mock_config.return_value = {
            'project_settings': {'projects_directory': '/tmp/projects'},
            'llm_providers': {},
            'ingestion_config': {}
        }
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_active_project.return_value = ("my_project", "/tmp/projects/my_project")
        mock_pm_instance.get_project_path_by_name.return_value = "/tmp/projects/my_project"
        mock_pm.return_value = mock_pm_instance

        session = M3Session()
        self.assertEqual(session.active_project_name, "my_project")

    @patch('core.session_manager.AnalyzeManager')
    @patch('core.session_manager.VectorManager')
    @patch('core.session_manager.PluginManager')
    @patch('core.session_manager.LLMManager')
    @patch('core.session_manager.ProjectManager')
    @patch('core.session_manager.get_config')
    def test_get_project_prompt_with_project(self, mock_config, mock_pm, mock_llm,
                                              mock_plugin, mock_vm, mock_am):
        """Prompt should include project name when one is active."""
        mock_config.return_value = {
            'project_settings': {'projects_directory': '/tmp/projects'},
            'llm_providers': {},
            'ingestion_config': {}
        }
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_active_project.return_value = ("test", "/tmp/projects/test")
        mock_pm_instance.get_project_path_by_name.return_value = "/tmp/projects/test"
        mock_pm.return_value = mock_pm_instance

        session = M3Session()
        prompt = session.get_project_prompt()
        self.assertIn("test", prompt)
        self.assertIn("m3", prompt)

    @patch('core.session_manager.AnalyzeManager')
    @patch('core.session_manager.VectorManager')
    @patch('core.session_manager.PluginManager')
    @patch('core.session_manager.LLMManager')
    @patch('core.session_manager.ProjectManager')
    @patch('core.session_manager.get_config')
    def test_get_project_prompt_no_project(self, mock_config, mock_pm, mock_llm,
                                            mock_plugin, mock_vm, mock_am):
        """Prompt should be generic when no project is active."""
        mock_config.return_value = {
            'project_settings': {'projects_directory': '/tmp/projects'},
            'llm_providers': {},
            'ingestion_config': {}
        }
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_active_project.return_value = (None, None)
        mock_pm.return_value = mock_pm_instance

        session = M3Session()
        prompt = session.get_project_prompt()
        self.assertEqual(prompt, "[m3]> ")

    @patch('core.session_manager.AnalyzeManager')
    @patch('core.session_manager.VectorManager')
    @patch('core.session_manager.PluginManager')
    @patch('core.session_manager.LLMManager')
    @patch('core.session_manager.ProjectManager')
    @patch('core.session_manager.get_config')
    def test_load_project_none_clears_state(self, mock_config, mock_pm, mock_llm,
                                             mock_plugin, mock_vm, mock_am):
        """Loading None should clear the active project."""
        mock_config.return_value = {
            'project_settings': {'projects_directory': '/tmp/projects'},
            'llm_providers': {},
            'ingestion_config': {}
        }
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_active_project.return_value = (None, None)
        mock_pm.return_value = mock_pm_instance

        session = M3Session()
        session.load_project(None)

        self.assertIsNone(session.active_project_name)
        self.assertIsNone(session.vector_manager)
        self.assertIsNone(session.analyze_manager)


class TestSessionManagerImport(unittest.TestCase):
    """Reports whether session_manager can be imported."""
    def test_import_status(self):
        if not SESSION_IMPORT_OK:
            self.skipTest(f"session_manager not importable: {SESSION_IMPORT_ERROR}")
        self.assertTrue(SESSION_IMPORT_OK)


if __name__ == '__main__':
    unittest.main()
