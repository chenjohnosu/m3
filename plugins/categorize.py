from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin
from core.prompt_manager import PromptManager

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")


class CategorizePlugin(LLMBaseAnalyzerPlugin):
    """
    Categorizes retrieved chunks based on user-defined options.
    """
    key: str = "categorize"
    description: str = "Categorizes chunks. Usage: ... run categorize \"<query>\" --options=\"cat1,cat2\""

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for categorization.
        """
        if not options:
            return PromptManager().get('plugin_categorize_error').format(query=query)

        return PromptManager().get('plugin_categorize').format(query=query, options=options)