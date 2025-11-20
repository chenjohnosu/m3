from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin
from core.prompt_manager import PromptManager

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar
    AnalyzeManager = TypeVar("AnalyzeManager")


class SummarizePlugin(LLMBaseAnalyzerPlugin):
    """
    Performs RAG-based summarization on a query.
    """
    key: str = "summarize"
    description: str = "Summarizes chunks. Usage: ... run summarize \"<query>\" --k 5 --threshold 0.7"

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for summarization.
        """
        return PromptManager().get('plugin_summarize').format(query=query)