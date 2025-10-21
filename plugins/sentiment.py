from plugins.llm_base_plugin import LLMBaseAnalyzerPlugin

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar
    AnalyzeManager = TypeVar("AnalyzeManager")


class SentimentPlugin(LLMBaseAnalyzerPlugin):
    """
    Performs sentiment analysis on chunks related to a query.
    """
    key: str = "sentiment"
    description: str = "Performs sentiment analysis on chunks relevant to a query."

    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Returns the system prompt for sentiment analysis.
        """
        return (
            "You are an expert sentiment analyst. "
            "Read the provided context chunks, which are all relevant to the user's query, "
            "and perform a sentiment analysis based *only* on that context.\n\n"
            f"USER QUERY: \"{query}\"\n\n"
            "First, provide an *overall* sentiment (Positive, Negative, Neutral, or Mixed) "
            "for the topic as a whole. \n\n"
            "Then, list any individual chunks that show particularly strong "
            "sentiment, explaining your reasoning."
        )