import click
from core.ingestion.base_pipeline import BasePipeline
from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
from core.llm_manager import LLMManager
# --- NEW: Import TextNode to create nodes ---
from llama_index.core.schema import TextNode
import hashlib


# --------------------------------------------

class CognitiveArchitectPipeline(BasePipeline):

    # --- MODIFIED: Accept llm_manager in the constructor ---
    def __init__(self, config, llm_manager: LLMManager):
        super().__init__(config)
        print("Initializing Cognitive Architect Pipeline...")
        # self.llm_manager = LLMManager(config) # <-- REMOVED: No longer create a new one
        self.llm_manager = llm_manager          # <-- ADDED: Use the persistent one
        # --- END MODIFIED ---

        self.cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        analysis_config = self.config.get('analysis_settings', {})
        self.embeddable_keys = analysis_config.get('metadata_keys_to_embed', ['themes'])
        click.echo(f"  > CogArc Pipeline: Embedding metadata keys = {self.embeddable_keys}", err=True)

        self.stage_0 = CogArcStage0Stratify(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_0_model'])
        )
        self.stage_1 = CogArcStage1Structure(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_1_model'])
        )
        self.stage_2 = CogArcStage2Enrich(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_2_model'])
        )
        self.stage_3 = CogArcStage3Synthesis(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_3_model'])
        )

        self.current_file_metadata = {}

    def run(self, documents, doc_type):
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        if documents:
            self.current_file_metadata = documents[0].metadata.copy()
            self.current_file_metadata.pop('text', None)

        pipeline_data = {'documents': documents}

        if doc_type == 'interview':
            pipeline_data = self.stage_0.process(pipeline_data)
        else:
            print("Skipping Stage 0 (Q&A Stratification) for non-interview document.")
            pass

        if not pipeline_data.get('documents'):
            print("No content available for further processing.")
            return {}

        structured_data = self.stage_1.process(pipeline_data)
        enriched_data = self.stage_2.process(structured_data)
        synthesized_data = self.stage_3.process(enriched_data)

        final_nodes = synthesized_data.get("primary_nodes", [])

        if not final_nodes and structured_data.get('documents'):
            click.echo("  > Finalizing nodes from Stage 1 data...")
            final_nodes = self._create_nodes_from_docs(structured_data.get('documents'))

        if final_nodes:
            final_nodes = self._apply_and_prepare_nodes(final_nodes)
            synthesized_data["primary_nodes"] = final_nodes

        print("--- Cognitive Architect Pipeline Finished ---")
        return synthesized_data

    def _create_nodes_from_docs(self, docs):
        """Helper to convert LlamaIndex Documents to TextNodes."""
        nodes = []
        for doc in docs:
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata
            )
            nodes.append(node)
        return nodes

    def _apply_and_prepare_nodes(self, nodes: list):
        """
        Applies document-level metadata (like summary) to all nodes
        and constructs the final text to be embedded.
        """
        prepared_nodes = []
        for node in nodes:
            original_text = node.get_content()
            node.metadata['original_text'] = original_text

            node.metadata.update(self.current_file_metadata)

            node.metadata['hash'] = hashlib.md5(original_text.encode()).hexdigest()

            searchable_parts = [original_text]

            for key in self.embeddable_keys:
                if node.metadata.get(key):
                    value_str = str(node.metadata.get(key)).strip()
                    if value_str:
                        searchable_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")

            if self.cogarc_settings.get('include_summary_in_embedding', False):
                if "holistic_summary" in node.metadata:
                    searchable_parts.append(f"Summary: {node.metadata['holistic_summary']}")

            node.set_content("\n\n".join(searchable_parts))

            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = list(node.metadata.keys())

            prepared_nodes.append(node)

        return prepared_nodes