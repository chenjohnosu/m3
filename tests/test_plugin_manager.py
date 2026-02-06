"""
Unit tests for core/plugin_manager.py
"""
import unittest
from core.plugin_manager import PluginManager
from plugins.base_plugin import BaseAnalyzerPlugin


class TestPluginManager(unittest.TestCase):
    def setUp(self):
        self.manager = PluginManager()

    def test_loads_plugins(self):
        plugins = self.manager.get_plugins()
        self.assertIsInstance(plugins, dict)
        self.assertGreater(len(plugins), 0)

    def test_known_plugins_loaded(self):
        plugins = self.manager.get_plugins()
        expected_keys = ['summarize', 'sentiment', 'entity', 'categorize',
                         'interpret', 'clustering', 'anomaly', 'visualize']
        for key in expected_keys:
            self.assertIn(key, plugins, f"Plugin '{key}' not found")

    def test_plugin_instances_are_base_plugin(self):
        plugins = self.manager.get_plugins()
        for key, plugin in plugins.items():
            self.assertIsInstance(plugin, BaseAnalyzerPlugin,
                                 f"Plugin '{key}' is not a BaseAnalyzerPlugin")

    def test_get_plugin_by_key(self):
        plugin = self.manager.get_plugin('summarize')
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.key, 'summarize')

    def test_get_plugin_nonexistent(self):
        plugin = self.manager.get_plugin('nonexistent_plugin_xyz')
        self.assertIsNone(plugin)

    def test_plugins_have_description(self):
        plugins = self.manager.get_plugins()
        for key, plugin in plugins.items():
            self.assertTrue(hasattr(plugin, 'description'),
                            f"Plugin '{key}' missing description")
            self.assertIsInstance(plugin.description, str)
            self.assertGreater(len(plugin.description), 0)

    def test_plugins_have_key(self):
        plugins = self.manager.get_plugins()
        for key, plugin in plugins.items():
            self.assertEqual(key, plugin.key)

    def test_no_duplicate_keys(self):
        plugins = self.manager.get_plugins()
        keys = list(plugins.keys())
        self.assertEqual(len(keys), len(set(keys)), "Duplicate plugin keys found")


if __name__ == '__main__':
    unittest.main()
