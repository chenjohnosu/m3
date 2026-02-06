"""
Unit tests for core/project_manager.py
"""
import unittest
import tempfile
import os
import shutil
from unittest.mock import patch
import utils.config as config_module


class TestProjectManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.projects_dir = os.path.join(self.temp_dir, "projects")
        os.makedirs(self.projects_dir, exist_ok=True)

        self.fake_config = {
            'project_settings': {'projects_directory': self.projects_dir},
            'llm_providers': {},
            'ingestion_config': {'known_doc_types': ['document'], 'default_doc_type': 'document'}
        }
        # Save original cache and replace it
        self._original_config = config_module._config
        config_module._config = self.fake_config

        from core.project_manager import ProjectManager
        self.manager = ProjectManager()
        # Verify our manager is pointing at our temp directory
        self.assertEqual(self.manager.projects_dir, self.projects_dir)

    def tearDown(self):
        config_module._config = self._original_config
        shutil.rmtree(self.temp_dir)

    def test_init_project_creates_directory(self):
        path, message = self.manager.init_project("test_project")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isdir(path))
        self.assertTrue(os.path.isdir(os.path.join(path, "corpus")))

    def test_init_project_duplicate_fails(self):
        self.manager.init_project("dup_project")
        path, message = self.manager.init_project("dup_project")
        self.assertIsNone(path)
        self.assertIn("already exists", message)

    def test_list_projects_empty(self):
        projects = self.manager.list_projects()
        self.assertEqual(projects, [])

    def test_list_projects_returns_created(self):
        self.manager.init_project("proj_a")
        self.manager.init_project("proj_b")
        projects = self.manager.list_projects()
        self.assertIn("proj_a", projects)
        self.assertIn("proj_b", projects)

    def test_set_active_project(self):
        self.manager.init_project("active_test")
        success, message = self.manager.set_active_project("active_test")
        self.assertTrue(success)

    def test_set_active_project_nonexistent(self):
        success, message = self.manager.set_active_project("ghost_project")
        self.assertFalse(success)

    def test_get_active_project(self):
        self.manager.init_project("get_active_test")
        self.manager.set_active_project("get_active_test")
        name, path = self.manager.get_active_project()
        self.assertEqual(name, "get_active_test")
        self.assertIsNotNone(path)

    def test_get_active_project_when_none_set(self):
        # Remove the active project file if it exists
        if os.path.exists(self.manager.active_project_file):
            os.remove(self.manager.active_project_file)
        name, path = self.manager.get_active_project()
        self.assertIsNone(name)
        self.assertIsNone(path)

    def test_get_project_path_by_name(self):
        self.manager.init_project("path_test")
        path = self.manager.get_project_path_by_name("path_test")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isdir(path))

    def test_get_project_path_by_name_nonexistent(self):
        path = self.manager.get_project_path_by_name("nonexistent")
        self.assertIsNone(path)

    def test_remove_project(self):
        self.manager.init_project("remove_test")
        success, message = self.manager.remove_project("remove_test")
        self.assertTrue(success)
        self.assertFalse(os.path.isdir(os.path.join(self.projects_dir, "remove_test")))

    def test_remove_project_nonexistent(self):
        success, message = self.manager.remove_project("ghost")
        self.assertFalse(success)

    def test_remove_active_project_clears_active(self):
        self.manager.init_project("active_remove")
        self.manager.set_active_project("active_remove")
        self.manager.remove_project("active_remove")
        name, path = self.manager.get_active_project()
        self.assertIsNone(name)


if __name__ == '__main__':
    unittest.main()
