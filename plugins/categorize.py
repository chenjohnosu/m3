from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin

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
            return (
                "You are a text categorization assistant. The user failed to provide "
                "categories. Please tell them to use the --options flag.\n\n"
                "Example: /a run categorize \"feedback\" --options='Positive,Negative,Neutral'"
            )

        return (
            "You are an expert text categorization assistant. "
            "Analyze the provided context chunks and categorize each chunk "
            f"in relation to the user's query: \"{query}\"\n\n"
            f"Use *only* the following categories: {options}\n\n"
            "List each chunk's source file and its assigned category. "
            "Provide a brief justification for each categorization."
        )