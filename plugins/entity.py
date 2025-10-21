from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin

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
            return (
                "You are an entity extraction assistant. The user failed to provide "
                "entity types to extract. Please tell them to use the --options flag.\n\n"
                "Example: /a run entity \"safety concerns\" --options='People,Locations,Equipment'"
            )

        return (
            "You are an expert entity extraction assistant. "
            "Read the provided context chunks and extract all entities of the "
            f"following types: {options}\n\n"
            "The analysis should focus on information relevant to the user's query: "
            f"\"{query}\"\n\n"
            "List the extracted entities, grouped by type. Only list entities "
            "explicitly found in the text."
        )