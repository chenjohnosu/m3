# core/ingestion/stages/cogarc_stage_1_structure.py

import json
from typing import List, Dict, Any
import click
from llama_index.core.schema import Document, BaseNode
# NEW: Import ChatMessage and MessageRole
from llama_index.core.llms import ChatMessage, MessageRole
from .base_stage import BaseStage


# REMOVED: from core.llm_dialogue import LLMDialogue

class CogArcStage1Structure(BaseStage):
    """
    Implements the 'Scaffolding' stage (Stage 1) of the CogArc pipeline.
    (3-phase implementation: Open Coding, Axial Coding, Tagging)
    """

    def __init__(self, llm_manager, db_manager, vector_manager, project_manager):
        super().__init__(llm_manager, db_manager, vector_manager, project_manager)
        self.stage_name = "Stage 1: Structure (Scaffolding)"

    def _get_llm(self, model_key='synthesis_model'):
        """Helper to get a specific LLM."""
        # 'synthesis_model' is used as a default, as it's seen in llm_manager.py
        # You can change 'synthesis_model' to any key from your config.yaml
        try:
            return self.llm_manager.get_llm(model_key)
        except ValueError:
            click.echo(f"  Warning: LLM model key '{model_key}' not found. Falling back to 'default_model'.", err=True)
            try:
                return self.llm_manager.get_llm('default_model')
            except ValueError as e:
                click.echo(f"  Fatal Error: 'default_model' also not found. {e}", err=True)
                raise

    def _perform_open_coding(self, chunk: BaseNode) -> List[str]:
        """
        Generates granular 'open codes' for a single chunk.
        """
        system_prompt = (
            "You are a qualitative researcher performing 'open coding'. "
            "Read the following text chunk and generate 3-5 granular, descriptive codes that capture the specific "
            "topics, actions, or concepts present. These codes should be short phrases (2-4 words)."
            "Respond ONLY with a valid JSON list of strings. "
            "Example: [\"fear of failure\", \"parental expectations\", \"future career anxiety\"]"
        )
        user_prompt = f"Text to code:\n---\n{chunk.get_content()}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        # Use a model appropriate for coding/generation
        llm = self._get_llm('default_model')
        click.echo("  > Calling LLM for open coding...")
        response = llm.chat(messages)
        response_content = response.message.content

        try:
            codes = json.loads(response_content)
            if isinstance(codes, list) and all(isinstance(c, str) for c in codes):
                chunk.metadata['open_codes'] = codes
                return codes
            else:
                click.echo(f"  Warning: LLM returned invalid JSON format for open codes. Storing raw response.",
                           err=True)
                chunk.metadata['open_codes'] = [response_content]
                return [response_content]
        except json.JSONDecodeError:
            click.echo(f"  Warning: Failed to decode JSON for open codes. Storing raw response.", err=True)
            chunk.metadata['open_codes'] = [response_content]
            return [response_content]

    def _perform_axial_coding(self, all_codes: List[str]) -> Dict[str, Any]:
        """
        Performs 'axial coding' by clustering a list of open codes into a hierarchical framework.
        """
        system_prompt = (
            "You are a senior qualitative researcher. You will be given a large list of 'open codes' "
            "generated from a corpus of documents. Your task is to perform 'axial coding' by clustering "
            "these codes into 5-7 high-level 'Core Themes'. Under each Core Theme, group related codes "
            "into 'Axial Categories'.\n"
            "Respond ONLY with a single JSON object in the following format:\n"
            "{\n"
            "  \"Core Theme 1\": {\n"
            "    \"Axial Category 1.1\": [\"code a\", \"code b\", ...],\n"
            "    \"Axial Category 1.2\": [\"code c\", \"code d\", ...]\n"
            "  },\n"
            "  \"Core Theme 2\": {\n"
            "    \"Axial Category 2.1\": [\"code e\", \"code f\", ...]\n"
            "  }\n"
            "}"
        )
        codes_json_string = json.dumps(list(set(all_codes)), indent=2)
        user_prompt = f"Here is the complete list of open codes to cluster:\n{codes_json_string}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        # Use a powerful model for this complex, single-shot corpus-level task
        llm = self._get_llm('synthesis_model')
        click.echo("  > Calling LLM for axial coding (framework generation)...")
        response = llm.chat(messages)
        response_content = response.message.content

        try:
            framework = json.loads(response_content)
            return framework
        except json.JSONDecodeError:
            click.echo(f"  Fatal Error: Failed to decode JSON for thematic framework. Aborting Stage 1.", err=True)
            return {"error": "Failed to generate framework", "raw_response": response_content}

    def _apply_thematic_tags(self, chunk: BaseNode, framework: Dict[str, Any]):
        """
        Applies the new thematic framework back to an individual chunk based on its open codes.
        """
        if 'open_codes' not in chunk.metadata:
            return  # Skip chunks that failed open coding

        system_prompt = (
            "You are a research assistant. You will be given a thematic framework and a list of open codes for a text chunk. "
            "Your job is to identify which 'Axial Categories' from the framework best apply to the chunk, based on its open codes. "
            "Respond ONLY with a JSON object containing two keys: 'axial_categories' (a list of strings) and 'core_themes' (a list of strings)."
            "Example: {\"axial_categories\": [\"Parental Expectations\", \"Academic Stress\"], \"core_themes\": [\"Family Influence\", \"Education\"]}"
        )

        framework_string = json.dumps(framework, indent=2)
        codes_string = json.dumps(chunk.metadata['open_codes'])

        user_prompt = (
            f"THEMATIC FRAMEWORK:\n{framework_string}\n\n"
            f"CHUNK'S OPEN CODES:\n{codes_string}\n\n"
            "Which 'Axial Categories' and 'Core Themes' apply to this chunk?"
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        llm = self._get_llm('default_model')
        click.echo("  > Calling LLM for thematic tagging...")
        response = llm.chat(messages)
        response_content = response.message.content

        try:
            tags = json.loads(response_content)
            chunk.metadata['axial_categories'] = tags.get('axial_categories', [])
            chunk.metadata['core_themes'] = tags.get('core_themes', [])
        except json.JSONDecodeError:
            click.echo(f"  Warning: Failed to decode JSON for thematic tags.", err=True)
            chunk.metadata['axial_categories'] = []
            chunk.metadata['core_themes'] = []

    def run(self, documents: List[Document]) -> List[Document]:
        """
        Executes the 3-phase scaffolding process on the entire corpus.
        """
        click.echo(f"  Starting {self.stage_name}...")

        # --- Phase 1.1: Open Coding ---
        click.echo("    Phase 1.1: Performing Open Coding on all chunks...")
        all_chunks = [chunk for doc in documents for chunk in doc.chunks]
        if not all_chunks:
            click.echo("    No chunks found to process. Skipping Stage 1.")
            return documents

        all_open_codes = set()
        with click.progressbar(all_chunks, label="    Open Coding") as bar:
            for chunk in bar:
                try:
                    codes = self._perform_open_coding(chunk)
                    all_open_codes.update(codes)
                except Exception as e:
                    click.echo(f"  Warning: Failed open coding for chunk {chunk.metadata.get('chunk_id')}: {e}",
                               err=True)

        click.echo(f"    Generated {len(all_open_codes)} unique open codes.")

        # --- Phase 1.2: Axial Coding ---
        click.echo("    Phase 1.2: Performing Axial Coding (Building Framework)...")
        try:
            thematic_framework = self._perform_axial_coding(list(all_open_codes))
            if "error" in thematic_framework:
                click.echo(f"    Failed to build thematic framework. Skipping tagging.", err=True)
                return documents

            self.db_manager.save_thematic_framework(thematic_framework)
            click.echo("    Thematic framework built and saved.")
        except Exception as e:
            click.echo(f"    Fatal Error during axial coding: {e}. Aborting Stage 1.", err=True)
            return documents

        # --- Phase 1.3: Thematic Tagging ---
        click.echo("    Phase 1.3: Applying new thematic tags to all chunks...")
        with click.progressbar(all_chunks, label="    Applying Tags") as bar:
            for chunk in bar:
                try:
                    self._apply_thematic_tags(chunk, thematic_framework)
                except Exception as e:
                    click.echo(f"  Warning: Failed to apply tags for chunk {chunk.metadata.get('chunk_id')}: {e}",
                               err=True)

        click.echo(f"  {self.stage_name} complete.")
        return documents