import click
import textwrap
from abc import abstractmethod
from plugins.base_plugin import BaseAnalyzerPlugin
from llama_index.core.llms import ChatMessage

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")


class LLMBaseAnalyzerPlugin(BaseAnalyzerPlugin):
    """
    Abstract base class for plugins that retrieve context and call an LLM.
    """

    # Use the powerful 'synthesis_model' by default for analysis
    model_key: str = "synthesis_model"

    @abstractmethod
    def get_system_prompt(self, query: str, options: str | None) -> str:
        """
        Plugins must implement this to return their specific system prompt.

        Args:
            query: The user's query text.
            options: The user's --options flag text.

        Returns:
            A string for the LLM system prompt.
        """
        pass

    def analyze(self, manager: "AnalyzeManager", **kwargs):
        """
        The main analysis loop: retrieve, build context, call LLM.
        """
        query = kwargs.get('query_text')
        k = kwargs.get('k', 5)  # Default to 5 chunks
        options = kwargs.get('options')

        if not query:
            click.secho(f"🔥 Error: The '{self.key}' plugin requires a query.", fg="red")
            click.echo(f"  > Usage: /a run {self.key} \"your query text\"")
            return

        click.secho(f"==> Running: {self.key} Plugin", fg="cyan")
        click.echo(f"  > Query: {query}")
        if options:
            click.echo(f"  > Options: {options}")
        click.echo(f"  > Retrieving Top {k} chunks...")

        # 1. Get LLM from manager
        try:
            # Use the manager's get_llm method
            llm = manager.get_llm(self.model_key)
        except Exception as e:
            click.secho(f"🔥 Error: Could not load LLM '{self.model_key}'. Check config.yaml.", fg="red")
            click.secho(f"  > Details: {e}", fg="red")
            return

        # 2. Retrieve chunks (like /a topk)
        retriever = manager.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)

        if not results:
            click.secho(f"  > No relevant chunks found for query.", fg="yellow")
            return

        # 3. Build context string
        context_parts = []
        sources = set()
        for i, res in enumerate(results):
            meta = res.node.metadata
            filename = meta.get('original_filename', 'Unknown')
            sources.add(filename)
            # Use 'original_text' from metadata for clean context
            text = meta.get('original_text', res.node.get_content())
            context_parts.append(f"--- Context Chunk {i + 1} (Source: {filename}) ---\n{text}")

        context_str = "\n\n".join(context_parts)

        # 4. Get formatted system prompt from subclass
        system_prompt = self.get_system_prompt(query, options)
        user_prompt = f"Here is the context to analyze:\n\n{context_str}"

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]

        # 5. Call LLM and print response
        click.echo(f"  > Sending context from {len(results)} chunks ({len(sources)} sources) to LLM...")
        try:
            response = llm.chat(messages)
            response_text = response.message.content

            click.secho(f"\n==> {self.key.title()} Analysis Complete:", bold=True)
            wrapped_text = textwrap.fill(
                response_text,
                width=100,
                initial_indent="  ",
                subsequent_indent="  "
            )
            click.echo(wrapped_text)

        except Exception as e:
            click.secho(f"🔥 Error during LLM analysis: {e}", fg="red")