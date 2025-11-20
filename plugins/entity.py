from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin
from core.prompt_manager import PromptManager

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar
    AnalyzeManager = TypeVar("AnalyzeManager")


class EntityPlugin(LLMBaseAnalyzerPlugin):
    """
    Extracts entities from chunks based on user-defined options.
    """
    key: str = "entity"
    description: str = "Extracts entities. Usage: ... run entity \"<query>\" --options=\"people,places\""

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for entity extraction.
        """
        if not options:
            return PromptManager().get('plugin_entity_error').format(query=query)

        return PromptManager().get('plugin_entity').format(query=query, options=options)