import click
from llama_index.core.llms import ChatMessage
from core.ingestion.stages.base_stage import BaseStage

SYSTEM_PROMPT = """
You are an expert qualitative data analyst. Your task is to synthesize a collection of text chunks from a single document into a concise, abstractive summary.
Analyze the provided text and perform the following actions:
1.  Read all the text chunks to understand the document's main points, arguments, and narrative flow.
2.  Generate a single, holistic summary (3-5 sentences) that captures the core essence and key takeaways of the entire document.
3.  Ensure the summary is abstractive, meaning you should synthesize ideas in your own words rather than just extracting and combining sentences.
hr
Important Rules:
-   The final output should be only the summary text, with no explanations, conversational text, or preamble like "Here is the summary:".
-   Focus on the overarching themes and conclusions from the text.
"""

class CogArcStage3Synthesis(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 3: Holistic Synthesis using LLM: {self.llm.model}")

        primary_nodes = data.get('primary_nodes')
        if not primary_nodes:
            print("  > No nodes to synthesize for Stage 3.")
            return data

        try:
            # --- FULL LLM IMPLEMENTATION ---
            # 1. Combine the content of all nodes into a single text block.
            full_text = "\n\n---\n\n".join([node.get_content() for node in primary_nodes])
            click.echo(f"  > Synthesizing content from {len(primary_nodes)} chunks...")

            # 2. Prepare the messages for the LLM.
            messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=full_text)
            ]

            # 3. Call the LLM to generate the holistic summary.
            response = self.llm.chat(messages)
            holistic_summary = response.message.content.strip()

            if not holistic_summary:
                raise ValueError("LLM returned an empty summary.")

            click.secho(f"  > Generated Summary: {holistic_summary}", fg="green")

            # 4. Write the generated summary back to the metadata of each node.
            for node in primary_nodes:
                node.metadata['holistic_summary'] = holistic_summary

            print(f"  > Successfully added holistic summary to {len(primary_nodes)} nodes.")

        except Exception as e:
            click.secho(f"  > Warning: Could not generate holistic summary. Reason: {e}", fg="yellow")
            # If synthesis fails, we still pass the data through without the summary.
            pass

        return data