from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin
from core.prompt_manager import PromptManager

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar
    AnalyzeManager = TypeVar("AnalyzeManager")


class SentimentPlugin(LLMBaseAnalyzerPlugin):
    """
    Performs sentiment analysis on chunks related to a query.
    """
    key: str = "sentiment"
    description: str = "Performs sentiment analysis. Usage: ... run sentiment \"<query>\" --threshold 0.7"

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for sentiment analysis.
        """
        return PromptManager().get('plugin_sentiment').format(query=query)