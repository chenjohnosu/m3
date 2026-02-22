import click
from core.prompt_manager import PromptManager
from llama_index.core.llms import ChatMessage
from core.ingestion.stages.base_stage import BaseStage

class CogArcStage3Synthesis(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 3: Holistic Synthesis using LLM: {self.llm.model}")

        primary_nodes = data.get('primary_nodes')
        if not primary_nodes:
            print("  > No nodes to synthesize for Stage 3.")
            return data

        system_prompt = PromptManager().get('ingestion_synthesis')

        try:
            # --- FULL LLM IMPLEMENTATION ---
            # 1. Combine the content of all nodes into a single text block.
            full_text = "\n\n---\n\n".join([node.get_content() for node in primary_nodes])
            click.echo(f"  > Synthesizing content from {len(primary_nodes)} chunks...")

            # 2. Prepare the messages for the LLM.
            messages = [
                ChatMessage(role="system", content=system_prompt),
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