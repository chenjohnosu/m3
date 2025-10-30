#
# plugins/interpret.py
#
import click
import textwrap
from plugins.base_plugin import BaseAnalyzerPlugin
from llama_index.core.llms import ChatMessage

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")

# System prompt for the meta-summary
SYSTEM_PROMPT = """
You are an expert qualitative data analyst. You will be provided with a list of individual document summaries from a research corpus.
Your task is to read all of them and synthesize them into a single, overarching "meta-summary" (3-5 paragraphs) that describes the entire collection as a whole.
Identify the key, high-level themes, patterns, and any potential contradictions that emerge from the corpus.
Do not just list the summaries; synthesize them.
"""


class InterpretPlugin(BaseAnalyzerPlugin):
    """
    Synthesizes all document summaries into a "meta-summary".
    """
    key: str = "interpret"
    description: str = "Synthesizes all document summaries into a meta-summary. Usage: /a run interpret"

    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        Gathers all unique holistic summaries and uses an LLM
        to synthesize them into a single meta-summary.
        """
        click.secho(f"==> Running: {self.key} Plugin (Corpus Meta-Summary)", fg="cyan")

        # 1. Get LLM from manager (use the powerful synthesis model)
        try:
            llm = manager.get_llm('synthesis_model')
        except Exception as e:
            click.secho(f"🔥 Error: Could not load LLM 'synthesis_model'. Check config.yaml.", fg="red")
            click.secho(f"  > Details: {e}", fg="red")
            return

        # 2. Get All Metadata from Vector Store
        try:
            # We only need 'metadatas' for this task
            all_data = manager.collection.get(
                include=["metadatas"]
            )
        except Exception as e:
            click.secho(f"🔥 Error: Could not retrieve data from vector store: {e}", fg="red")
            return

        metadatas = all_data.get('metadatas')
        if not metadatas:
            click.secho("  > No data found in vector store.", fg="yellow")
            return

        # 3. De-duplicate summaries
        # As you noted, each file's summary is on all its chunks.
        # A 'set' will automatically handle de-duplication.
        unique_summaries = set()
        for meta in metadatas:
            summary = meta.get('holistic_summary')
            if summary:
                unique_summaries.add(summary)

        if not unique_summaries:
            click.secho("  > No 'holistic_summary' metadata found in any chunks.", fg="yellow")
            click.secho("  > Try running '/corpus ingest' to generate summaries first.", fg="yellow")
            return

        click.echo(f"  > Found {len(unique_summaries)} unique document summaries to analyze.")
        click.echo("  > Sending to LLM for meta-synthesis...")

        # 4. Build context and call LLM
        context_parts = []
        for i, summary in enumerate(list(unique_summaries)):
            context_parts.append(f"--- Document Summary {i + 1} ---\n{summary}")

        context_str = "\n\n".join(context_parts)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=context_str)
        ]

        # 5. Call LLM and print response
        try:
            response = llm.chat(messages)
            response_text = response.message.content

            click.secho(f"\n==> Corpus Meta-Summary Complete:", bold=True)
            wrapped_text = textwrap.fill(
                response_text,
                width=100,
                initial_indent="  ",
                subsequent_indent="  "
            )
            click.echo(wrapped_text)

        except Exception as e:
            click.secho(f"🔥 Error during LLM analysis: {e}", fg="red")