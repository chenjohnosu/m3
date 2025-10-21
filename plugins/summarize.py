from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar
    AnalyzeManager = TypeVar("AnalyzeManager")


class SummarizePlugin(LLMBaseAnalyzerPlugin):
    """
    Performs RAG-based summarization on a query.
    """
    key: str = "summarize"
    description: str = "Summarizes relevant chunks based on a query."

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for summarization.
        """
        return (
            "You are an expert summarization assistant. "
            "Based *only* on the context provided by the user, write a concise, "
            "multi-paragraph summary that directly answers the user's query.\n\n"
            "Do not use any information other than the context provided.\n\n"
            f"USER QUERY: \"{query}\""
        )