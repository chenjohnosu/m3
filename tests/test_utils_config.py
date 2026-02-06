"""
Unit tests for utils/config.py
"""
import unittest
import os
import tempfile
import shutil
import yaml
from unittest.mock import patch

# We must patch the config module's global cache before importing
import utils.config as config_module


class TestGetConfigDir(unittest.TestCase):
    def test_returns_monkey3_dir(self):
        result = config_module.get_config_dir()
        self.assertTrue(result.endswith(".monkey3"))
        self.assertIn(os.path.expanduser("~"), result)


class TestGetConfigPath(unittest.TestCase):
    def test_returns_config_yaml_path(self):
        result = config_module.get_config_path()
        self.assertTrue(result.endswith("config.yaml"))
        self.assertIn(".monkey3", result)


class TestCreateDefaultConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_config_path = os.path.join(self.temp_dir, "config.yaml")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(config_module, 'get_config_path')
    def test_creates_config_when_missing(self, mock_path):
        mock_path.return_value = self.fake_config_path
        config_module.create_default_config_if_not_exists()
        self.assertTrue(os.path.exists(self.fake_config_path))

    @patch.object(config_module, 'get_config_path')
    def test_does_not_overwrite_existing_config(self, mock_path):
        # Write something to the path first
        with open(self.fake_config_path, 'w') as f:
            f.write("existing: true\n")
        mock_path.return_value = self.fake_config_path

        config_module.create_default_config_if_not_exists()
        with open(self.fake_config_path, 'r') as f:
            content = f.read()
        self.assertIn("existing: true", content)


class TestGetConfig(unittest.TestCase):
    def setUp(self):
        # Reset the global cache
        config_module._config = None
        self.temp_dir = tempfile.mkdtemp()
        self.fake_config_path = os.path.join(self.temp_dir, "config.yaml")
        # Write a minimal valid config
        minimal_config = {
            'project_settings': {'projects_directory': os.path.join(self.temp_dir, 'projects')},
            'llm_providers': {},
            'ingestion_config': {'known_doc_types': ['document'], 'default_doc_type': 'document'}
        }
        with open(self.fake_config_path, 'w') as f:
            yaml.dump(minimal_config, f)

    def tearDown(self):
        config_module._config = None
        shutil.rmtree(self.temp_dir)

    @patch.object(config_module, 'get_config_path')
    @patch.object(config_module, 'create_default_config_if_not_exists')
    def test_returns_dict(self, mock_create, mock_path):
        mock_path.return_value = self.fake_config_path
        result = config_module.get_config()
        self.assertIsInstance(result, dict)
        self.assertIn('project_settings', result)

    @patch.object(config_module, 'get_config_path')
    @patch.object(config_module, 'create_default_config_if_not_exists')
    def test_caches_result(self, mock_create, mock_path):
        mock_path.return_value = self.fake_config_path
        result1 = config_module.get_config()
        result2 = config_module.get_config()
        self.assertIs(result1, result2)


if __name__ == '__main__':
    unittest.main()
