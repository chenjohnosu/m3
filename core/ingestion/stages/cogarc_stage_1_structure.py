import click
import json
import re
from core.ingestion.stages.base_stage import BaseStage
from llama_index.core.llms import ChatMessage

# A new system prompt to guide the LLM in identifying high-level themes.
SYSTEM_PROMPT = """
You are an expert qualitative data analyst. Your task is to read a piece of text and identify its core underlying themes or topics.
Analyze the provided text and perform the following actions:
1.  Read the text to understand its main points and arguments.
2.  Identify 2-4 distinct, high-level themes that capture the essence of the text. A theme should be a short phrase (3-5 words).
3.  Structure your output as a single, valid JSON array of strings.
4.  Each string in the array must be a single theme.

Example Output: `["Online learning experiences", "Connection to the university", "Career preparation and skills"]`

Important Rules:
-   Focus on the conceptual topics, not just keywords.
-   Ensure the final output is only the JSON array, with no explanations or conversational text.
"""


class CogArcStage1Structure(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 1: Thematic Scaffolding using LLM: {self.llm.model}")
        docs_to_process = data.get('documents', [])
        if not docs_to_process:
            print("  > No documents to process for Stage 1.")
            return data

        structured_docs = []
        for doc in docs_to_process:
            try:
                # Avoid processing very short texts that likely lack thematic depth.
                if len(doc.text.split()) < 25:
                    # --- THIS IS THE FIX ---
                    # Clean up the text for a single-line log message
                    cleaned_text = doc.text.replace('\n', ' ').strip()
                    click.secho(
                        f"  > Skipping thematic analysis for short text chunk from '{doc.metadata.get('original_filename', 'Unknown')}'.",
                        fg="yellow")
                    click.secho(f"    > Skipped Text: \"{cleaned_text}\"", fg="yellow")
                    # --- END OF FIX ---

                    structured_docs.append(doc)
                    continue

                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=doc.text)
                ]

                click.echo(f"  > Analyzing themes for chunk from '{doc.metadata.get('original_filename', 'Unknown')}'.")
                response = self.llm.chat(messages)
                response_text = response.message.content

                # Use the robust regex method to extract the JSON array of themes.
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON array of themes found in the LLM response.")

                themes = json.loads(json_match.group(0))

                if themes and all(isinstance(t, str) for t in themes):
                    # --- THE FIX ---
                    # Convert the list of themes into a single, comma-separated string.
                    themes_str = ", ".join(themes)
                    doc.metadata['themes'] = themes_str
                    click.secho(f"    > Identified themes: {themes_str}", fg="blue")
                else:
                    raise ValueError("LLM response was not a valid list of strings.")

                structured_docs.append(doc)

            except (json.JSONDecodeError, ValueError) as e:
                click.secho(f"  > Warning: Could not extract themes for chunk. Reason: {e}", fg="yellow")
                # If analysis fails, pass the original document through without theme metadata.
                structured_docs.append(doc)
            except Exception as e:
                click.secho(f"  > An unexpected error occurred during structuring: {e}", fg="red")
                structured_docs.append(doc)

        # Pass the documents, now enriched with theme metadata, to the next stage.
        data['documents'] = structured_docs
        print(f"  > Completed thematic analysis for {len(docs_to_process)} documents.")
        return data