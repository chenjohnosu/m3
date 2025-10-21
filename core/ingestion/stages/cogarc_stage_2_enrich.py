# core/ingestion/stages/cogarc_stage_2_enrich.py

import json
from typing import List
import click
from llama_index.core.schema import Document, BaseNode
# NEW: Import ChatMessage and MessageRole
from llama_index.core.llms import ChatMessage, MessageRole
from .base_stage import BaseStage


# REMOVED: from core.llm_dialogue import LLMDialogue

class CogArcStage2Enrich(BaseStage):
    """
    Implements the 'Enrichment' stage (Stage 2) of the CogArc pipeline.

    This stage enriches each chunk (BaseNode) with additional metadata:
    1.  Generates a hypothetical question the chunk could answer.
    2.  Performs nuanced affective (sentiment) analysis.
    3.  Identifies potential paraphrases/semantic duplicates from the *existing* vector store.
    """

    def __init__(self, llm_manager, db_manager, vector_manager, project_manager):
        super().__init__(llm_manager, db_manager, vector_manager, project_manager)
        self.stage_name = "Stage 2: Enrich"

    def _get_llm(self, model_key='default_model'):
        """Helper to get a specific LLM."""
        # Use 'default_model' for these simpler tasks
        try:
            return self.llm_manager.get_llm(model_key)
        except ValueError:
            click.echo(f"  Warning: LLM model key '{model_key}' not found. Falling back to 'synthesis_model'.",
                       err=True)
            try:
                return self.llm_manager.get_llm('synthesis_model')
            except ValueError as e:
                click.echo(f"  Fatal Error: 'synthesis_model' also not found. {e}", err=True)
                raise

    def _generate_hypothetical_question(self, chunk: BaseNode):
        """
        Generates a hypothetical question that the chunk's content could answer.
        """
        system_prompt = (
            "You are a research assistant. Read the following text. "
            "Generate a single, concise, relevant question that this text could be the answer to. "
            "Respond ONLY with the question."
        )
        user_prompt = f"Text:\n---\n{chunk.get_content()}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        llm = self._get_llm()
        click.echo("  > Calling LLM for hypothetical question...")
        response = llm.chat(messages)
        chunk.metadata['hypothetical_question'] = response.message.content

    def _perform_affective_analysis(self, chunk: BaseNode):
        """
        Performs nuanced affective (sentiment) analysis on the chunk.
        """
        system_prompt = (
            "Analyze the affective content of the following text. "
            "Respond ONLY with a valid JSON object containing three keys:\n"
            "1. `primary_emotion`: The single strongest emotion (e.g., 'joy', 'sadness', 'anger', 'anxiety', 'frustration', 'neutral').\n"
            "2. `sentiment_score`: A float from -1.0 (highly negative) to 1.0 (highly positive).\n"
            "3. `emotion_justification`: A brief explanation (1-2 sentences) for your choices."
        )
        user_prompt = f"Text to analyze:\n---\n{chunk.get_content()}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        llm = self._get_llm()
        click.echo("  > Calling LLM for affective analysis...")
        response = llm.chat(messages)
        response_content = response.message.content

        try:
            analysis = json.loads(response_content)
            chunk.metadata['primary_emotion'] = analysis.get('primary_emotion', 'unknown')
            chunk.metadata['sentiment_score'] = analysis.get('sentiment_score', 0.0)
            chunk.metadata['emotion_justification'] = analysis.get('emotion_justification', 'N/A')
        except json.JSONDecodeError:
            click.echo(f"  Warning: Failed to decode JSON for affective analysis. Storing raw response.", err=True)
            chunk.metadata['primary_emotion'] = 'error'
            chunk.metadata['sentiment_score'] = 0.0
            chunk.metadata['emotion_justification'] = response_content

    def _identify_paraphrases(self, chunk: BaseNode, top_k: int = 5):
        """
        Identifies potential paraphrases by searching the vector store and verifying with an LLM.
        """
        current_chunk_id = chunk.metadata.get('chunk_id')
        if not current_chunk_id:
            return

        try:
            similar_chunks = self.vector_manager.search(chunk.get_content(), top_k=top_k)
        except Exception as e:
            click.echo(f"  Warning: Vector search failed for paraphrase ID: {e}", err=True)
            return

        system_prompt = (
            "You will be given two text chunks, Chunk A and Chunk B. "
            "Do they express the exact same core idea, just in different words? "
            "Answer ONLY with 'YES' or 'NO'."
        )

        llm = self._get_llm()

        for result_chunk, _ in similar_chunks:
            candidate_chunk_id = result_chunk.metadata.get('chunk_id')

            if not candidate_chunk_id or candidate_chunk_id == current_chunk_id:
                continue

            user_prompt = (
                f"Chunk A:\n{chunk.get_content()}\n\n"
                f"Chunk B:\n{result_chunk.get_content()}"
            )

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]

            click.echo("  > Calling LLM for paraphrase verification...")
            response = llm.chat(messages)
            response_content = response.message.content

            if "yes" in response_content.lower():
                self.db_manager.add_paraphrase_link(current_chunk_id, candidate_chunk_id)

    def run(self, documents: List[Document]) -> List[Document]:
        """
        Executes the enrichment process for each chunk in each document.
        """
        click.echo(f"  Starting {self.stage_name}...")

        all_chunks = [chunk for doc in documents for chunk in doc.chunks]

        if not all_chunks:
            click.echo("    No chunks found to enrich. Skipping Stage 2.")
            return documents

        click.echo("    Enriching chunks with Hypothetical Questions and Affective Analysis...")
        with click.progressbar(all_chunks, label="    Enriching") as bar:
            for chunk in bar:
                chunk_id = chunk.metadata.get('chunk_id', 'unknown_chunk')

                try:
                    self._generate_hypothetical_question(chunk)
                except Exception as e:
                    click.echo(f"    Warning (Chunk {chunk_id}): Failed to generate hypothetical question: {e}",
                               err=True)

                try:
                    self._perform_affective_analysis(chunk)
                except Exception as e:
                    click.echo(f"    Warning (Chunk {chunk_id}): Failed to perform affective analysis: {e}", err=True)

        click.echo("    Identifying paraphrases against existing vector store...")
        with click.progressbar(all_chunks, label="    Finding Paraphrases") as bar:
            for chunk in bar:
                chunk_id = chunk.metadata.get('chunk_id', 'unknown_chunk')
                try:
                    self._identify_paraphrases(chunk)
                except Exception as e:
                    click.echo(f"    Warning (Chunk {chunk_id}): Failed to identify paraphrases: {e}", err=True)

        click.echo(f"  {self.stage_name} complete.")
        return documents