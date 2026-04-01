import click
import json
import re
from core.prompt_manager import PromptManager
from core.ingestion.stages.base_stage import BaseStage

class CogArcStage1Structure(BaseStage):

    # This method must be defined to implement the abstract method from BaseStage
    def process(self, data):
        print(f"Executing CogArc Stage 1: Thematic Scaffolding using LLM: {self.llm.model}")
        docs_to_process = data.get('documents', [])
        if not docs_to_process:
            print("  > No documents to process for Stage 1.")
            return data

        system_prompt = PromptManager().get('ingestion_structure')

        structured_docs = []
        for doc in docs_to_process:
            try:
                # Avoid processing very short texts that likely lack thematic depth.
                if len(doc.text.split()) < 25:
                    cleaned_text = doc.text.replace('\n', ' ').strip()
                    click.secho(
                        f"  > Skipping thematic analysis for short text chunk from '{doc.metadata.get('original_filename', 'Unknown')}'.",
                        fg="yellow")
                    click.secho(f"    > Skipped Text: \"{cleaned_text}\"", fg="yellow")

                    structured_docs.append(doc)
                    continue

                from llama_index.core.llms import ChatMessage
                messages = [
                    ChatMessage(role="system", content=system_prompt),
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