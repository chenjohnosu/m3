import importlib
import inspect  # <-- Make sure 'inspect' is imported
from pathlib import Path
from plugins.base_plugin import BaseAnalyzerPlugin


class PluginManager:
    """
    Discovers and loads all analyzer plugins from the top-level /plugins directory.
    """

    def __init__(self):
        # Path(__file__) is this file (core/plugin_manager.py)
        # .parent is core/
        # .parent.parent is the project root
        self.plugins_path = Path(__file__).parent.parent / "plugins"
        self._plugins = {}
        self._load_plugins()

    def _load_plugins(self):
        """
        Scans the plugins directory and loads all classes that
        inherit from BaseAnalyzerPlugin.
        """
        for f in self.plugins_path.glob("*.py"):

            # --- FIX 1: Be more specific about files to skip ---
            # Skip __init__, base_, and llm_base_ files
            if f.name.startswith(("_", "base_", "llm_base_")):
                continue

            # The module name is now 'plugins.clustering_plugin', for example
            module_name = f"plugins.{f.stem}"
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):

                    # --- FIX 2: Add more robust checks ---
                    # 1. Must be a subclass of BaseAnalyzerPlugin
                    # 2. Must NOT be BaseAnalyzerPlugin itself
                    # 3. Must NOT be an abstract class (like LLMBaseAnalyzerPlugin)
                    # 4. Must be defined in this module (not imported)
                    if (issubclass(obj, BaseAnalyzerPlugin) and
                            obj != BaseAnalyzerPlugin and
                            not inspect.isabstract(obj) and
                            obj.__module__ == module_name):

                        plugin_instance = obj()
                        if plugin_instance.key in self._plugins:
                            raise ValueError(f"Duplicate plugin key found: {plugin_instance.key}")
                        self._plugins[plugin_instance.key] = plugin_instance

            except Exception as e:
                # This is the error message you were seeing
                print(f"Failed to load plugin {module_name}: {e}")

    def get_plugins(self):
        """Returns a dict of all loaded plugins {key: instance}."""
        return self._plugins

    def get_plugin(self, key: str):
        """Returns a single plugin instance by its key."""
        return self._plugins.get(key)