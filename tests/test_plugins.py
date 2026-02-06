"""
Unit tests for the plugin system:
  - plugins/base_plugin.py
  - plugins/llm_base_plugin.py
  - plugins/summarize.py, sentiment.py, entity.py, categorize.py
  - plugins/interpret.py
  - plugins/clustering.py
  - plugins/anomaly.py
  - plugins/visualize.py
"""
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from plugins.base_plugin import BaseAnalyzerPlugin
from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin


# ─────────────────────────────────────────────
# Base Plugin
# ─────────────────────────────────────────────

class TestBaseAnalyzerPlugin(unittest.TestCase):
    def test_is_abstract(self):
        """BaseAnalyzerPlugin should not be instantiable directly."""
        with self.assertRaises(TypeError):
            BaseAnalyzerPlugin()

    def test_has_required_attributes(self):
        self.assertTrue(hasattr(BaseAnalyzerPlugin, 'key'))
        self.assertTrue(hasattr(BaseAnalyzerPlugin, 'description'))
        self.assertTrue(hasattr(BaseAnalyzerPlugin, 'analyze'))


# ─────────────────────────────────────────────
# LLM Base Plugin
# ─────────────────────────────────────────────

class TestLLMBaseAnalyzerPlugin(unittest.TestCase):
    def test_is_abstract(self):
        """LLMBaseAnalyzerPlugin should not be instantiable directly."""
        with self.assertRaises(TypeError):
            LLMBaseAnalyzerPlugin()

    def test_has_model_key(self):
        self.assertTrue(hasattr(LLMBaseAnalyzerPlugin, 'model_key'))
        self.assertEqual(LLMBaseAnalyzerPlugin.model_key, 'synthesis_model')

    def test_has_get_system_prompt(self):
        self.assertTrue(hasattr(LLMBaseAnalyzerPlugin, 'get_system_prompt'))


# ─────────────────────────────────────────────
# Summarize Plugin
# ─────────────────────────────────────────────

class TestSummarizePlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.summarize import SummarizePlugin
        plugin = SummarizePlugin()
        self.assertEqual(plugin.key, 'summarize')
        self.assertIsInstance(plugin, BaseAnalyzerPlugin)

    def test_get_system_prompt(self):
        from plugins.summarize import SummarizePlugin
        plugin = SummarizePlugin()
        prompt = plugin.get_system_prompt("test query", None)
        self.assertIn("test query", prompt)
        self.assertIn("summar", prompt.lower())


# ─────────────────────────────────────────────
# Sentiment Plugin
# ─────────────────────────────────────────────

class TestSentimentPlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.sentiment import SentimentPlugin
        plugin = SentimentPlugin()
        self.assertEqual(plugin.key, 'sentiment')

    def test_get_system_prompt(self):
        from plugins.sentiment import SentimentPlugin
        plugin = SentimentPlugin()
        prompt = plugin.get_system_prompt("safety concerns", None)
        self.assertIn("safety concerns", prompt)
        self.assertIn("sentiment", prompt.lower())


# ─────────────────────────────────────────────
# Entity Plugin
# ─────────────────────────────────────────────

class TestEntityPlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.entity import EntityPlugin
        plugin = EntityPlugin()
        self.assertEqual(plugin.key, 'entity')

    def test_get_system_prompt_with_options(self):
        from plugins.entity import EntityPlugin
        plugin = EntityPlugin()
        prompt = plugin.get_system_prompt("query", "People,Places")
        self.assertIn("People,Places", prompt)

    def test_get_system_prompt_without_options(self):
        from plugins.entity import EntityPlugin
        plugin = EntityPlugin()
        prompt = plugin.get_system_prompt("query", None)
        self.assertIn("--options", prompt)


# ─────────────────────────────────────────────
# Categorize Plugin
# ─────────────────────────────────────────────

class TestCategorizePlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.categorize import CategorizePlugin
        plugin = CategorizePlugin()
        self.assertEqual(plugin.key, 'categorize')

    def test_get_system_prompt_with_options(self):
        from plugins.categorize import CategorizePlugin
        plugin = CategorizePlugin()
        prompt = plugin.get_system_prompt("feedback", "Positive,Negative")
        self.assertIn("Positive,Negative", prompt)

    def test_get_system_prompt_without_options(self):
        from plugins.categorize import CategorizePlugin
        plugin = CategorizePlugin()
        prompt = plugin.get_system_prompt("feedback", None)
        self.assertIn("--options", prompt)


# ─────────────────────────────────────────────
# Interpret Plugin
# ─────────────────────────────────────────────

class TestInterpretPlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.interpret import InterpretPlugin
        plugin = InterpretPlugin()
        self.assertEqual(plugin.key, 'interpret')
        self.assertIsInstance(plugin, BaseAnalyzerPlugin)

    def test_has_analyze_method(self):
        from plugins.interpret import InterpretPlugin
        plugin = InterpretPlugin()
        self.assertTrue(callable(getattr(plugin, 'analyze')))


# ─────────────────────────────────────────────
# Clustering Plugin
# ─────────────────────────────────────────────

class TestClusteringPlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.clustering import ClusteringPlugin
        plugin = ClusteringPlugin()
        self.assertEqual(plugin.key, 'clustering')
        self.assertIsInstance(plugin, BaseAnalyzerPlugin)

    def test_sklearn_availability_flag(self):
        from plugins.clustering import SKLEARN_AVAILABLE
        self.assertIsInstance(SKLEARN_AVAILABLE, bool)

    def test_has_analyze_method(self):
        from plugins.clustering import ClusteringPlugin
        plugin = ClusteringPlugin()
        self.assertTrue(callable(getattr(plugin, 'analyze')))


# ─────────────────────────────────────────────
# Anomaly Plugin
# ─────────────────────────────────────────────

class TestAnomalyPlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.anomaly import AnomalyPlugin
        plugin = AnomalyPlugin()
        self.assertEqual(plugin.key, 'anomaly')
        self.assertIsInstance(plugin, BaseAnalyzerPlugin)

    def test_sklearn_availability_flag(self):
        from plugins.anomaly import SKLEARN_AVAILABLE
        self.assertIsInstance(SKLEARN_AVAILABLE, bool)

    def test_has_analyze_method(self):
        from plugins.anomaly import AnomalyPlugin
        plugin = AnomalyPlugin()
        self.assertTrue(callable(getattr(plugin, 'analyze')))


# ─────────────────────────────────────────────
# Visualize Plugin
# ─────────────────────────────────────────────

class TestVisualizePlugin(unittest.TestCase):
    def test_instantiation(self):
        from plugins.visualize import VisualizePlugin
        plugin = VisualizePlugin()
        self.assertEqual(plugin.key, 'visualize')
        self.assertIsInstance(plugin, BaseAnalyzerPlugin)

    def test_viz_availability_flag(self):
        from plugins.visualize import VIZ_AVAILABLE
        self.assertIsInstance(VIZ_AVAILABLE, bool)

    def test_has_analyze_method(self):
        from plugins.visualize import VisualizePlugin
        plugin = VisualizePlugin()
        self.assertTrue(callable(getattr(plugin, 'analyze')))


if __name__ == '__main__':
    unittest.main()
