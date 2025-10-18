import click
from core.ingestion.stages.base_stage import BaseStage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
import json
import re

# A system prompt designed to generate a concise, relevant question for a chunk of text.
# --- UPDATED: Removed all examples to prevent "prompt bleed" ---
SYSTEM_PROMPT = """
You are an expert in synthesizing information. Your task is to read the following text chunk and generate a single, concise, and relevant question that this text could answer.
The question should be a natural-language query that a user might ask to find this specific information.

---
IMPORTANT RULES:
1.  Your output MUST be only the question itself.
2.  Do NOT include any preamble like "Here is the question:".
3.  Do NOT copy any part of this system prompt.
4.  Generate ONE question ONLY.
5.  The output must be a single string, not a JSON object.
---

Read the text below and provide only the question.
"""


class CogArcStage2Enrich(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 2: Micro-Context Enrichment using LLM: {self.llm.model}")

        documents_to_chunk = data.get('documents', [])
        if not documents_to_chunk:
            print("  > No documents to process for Stage 2.")
            return data

        # Use the SentenceSplitter to create the final text chunks (nodes).
        # We can configure this from config.yaml in a future update.
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents_to_chunk)

        enriched_nodes = []
        for i, node in enumerate(nodes):
            try:
                click.echo(f"  > Enriching chunk {i + 1}/{len(nodes)}...")

                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=node.get_content())
                ]

                response = self.llm.chat(messages)
                hypothetical_question = response.message.content.strip()

                # Basic cleaning of the question
                if hypothetical_question.startswith('"') and hypothetical_question.endswith('"'):
                    hypothetical_question = hypothetical_question[1:-1]

                if hypothetical_question:
                    # Add the new, specific question to this chunk's metadata.
                    # It inherits existing metadata like 'themes'.
                    node.metadata['hypothetical_question'] = hypothetical_question
                    click.secho(f"    > Generated Question: {hypothetical_question}", fg="magenta")

                enriched_nodes.append(node)

            except Exception as e:
                click.secho(f"  > Warning: Could not generate hypothetical question for chunk {i + 1}. Reason: {e}",
                            fg="yellow")
                # If enrichment fails, still include the original node.
                enriched_nodes.append(node)

        # Assign the final, enriched chunks to 'primary_nodes' for the vector store.
        data['primary_nodes'] = enriched_nodes

        print(f"  > Generated and enriched {len(enriched_nodes)} text chunks.")
        return data