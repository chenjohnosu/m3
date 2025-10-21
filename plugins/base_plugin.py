import click
from abc import ABC, abstractmethod

# Forward-declare AnalyzeManager to avoid circular import issues
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")


class BaseAnalyzerPlugin(ABC):
    """
    Abstract base class for an M3 analyzer plugin.
    """

    # A unique key for the plugin, used in the CLI
    # e.g., /analyze run <key>
    key: str = "base"

    # A short description for the help command
    description: str = "Base plugin"

    @abstractmethod
    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        The main execution method for the plugin.

        It receives an instance of the AnalyzeManager, which gives it
        access to the vector store, index, and collection via:
        - manager.index
        - manager.collection
        - manager.embed_model
        """
        pass