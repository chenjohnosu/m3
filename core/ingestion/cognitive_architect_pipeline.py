import click
from core.ingestion.base_pipeline import BasePipeline  # <-- Corrected from base_pipeline.py
from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
from core.llm_manager import LLMManager
# --- NEW: Import TextNode to create nodes ---
from llama_index.core.schema import TextNode
import hashlib


# --------------------------------------------

class CognitiveArchitectPipeline(BasePipeline):  # <-- Corrected from BasePipeline
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Cognitive Architect Pipeline...")
        llm_manager = LLMManager(config)
        self.cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        # --- MODIFIED: Load embeddable keys from config ---
        analysis_config = self.config.get('analysis_settings', {})
        self.embeddable_keys = analysis_config.get('metadata_keys_to_embed', ['themes'])  # Default to 'themes'
        click.echo(f"  > CogArc Pipeline: Embedding metadata keys = {self.embeddable_keys}", err=True)
        # --- End Modification ---

        self.stage_0 = CogArcStage0Stratify(
            config, llm=llm_manager.get_llm(self.cogarc_settings['stage_0_model'])
        )
        self.stage_1 = CogArcStage1Structure(
            config, llm=llm_manager.get_llm(self.cogarc_settings['stage_1_model'])
        )
        self.stage_2 = CogArcStage2Enrich(
            config, llm=llm_manager.get_llm(self.cogarc_settings['stage_2_model'])
        )
        self.stage_3 = CogArcStage3Synthesis(
            config, llm=llm_manager.get_llm(self.cogarc_settings['stage_3_model'])
        )

        # --- NEW: Store file metadata ---
        self.current_file_metadata = {}
        # --------------------------------

    def run(self, documents, doc_type):
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        # --- NEW: Store file-level metadata ---
        # All documents passed in are from the same file
        if documents:
            self.current_file_metadata = documents[0].metadata.copy()
            self.current_file_metadata.pop('text', None)  # Remove text, keep metadata
        # --------------------------------------

        pipeline_data = {'documents': documents}

        if doc_type == 'interview':
            pipeline_data = self.stage_0.process(pipeline_data)
        else:
            print("Skipping Stage 0 (Q&A Stratification) for non-interview document.")
            # --- NEW: Handle non-interview docs ---
            # Stage 0 creates 'documents' from text, but for non-interviews,
            # we need to ensure the text content is passed to Stage 1.
            # We assume stage 0's output is 'documents', so let's just pass it along.
            pass
            # --------------------------------------

        if not pipeline_data.get('documents'):
            print("No content available for further processing.")
            return {}

        structured_data = self.stage_1.process(pipeline_data)
        enriched_data = self.stage_2.process(structured_data)
        synthesized_data = self.stage_3.process(enriched_data)

        # --- MODIFIED: Apply final text formatting ---
        # This logic is now handled in _create_text_node
        # The output of stage 3 is 'primary_nodes'
        final_nodes = synthesized_data.get("primary_nodes", [])

        # --- NEW: Create nodes if not already created by Stage 2 ---
        # This handles the case where Stage 2 is skipped or fails
        if not final_nodes and structured_data.get('documents'):
            click.echo("  > Finalizing nodes from Stage 1 data...")
            final_nodes = self._create_nodes_from_docs(structured_data.get('documents'))

        # Apply file-level metadata and searchable text
        if final_nodes:
            final_nodes = self._apply_and_prepare_nodes(final_nodes)
            synthesized_data["primary_nodes"] = final_nodes
        # --- END MODIFICATION ---

        print("--- Cognitive Architect Pipeline Finished ---")
        return synthesized_data

    def _create_nodes_from_docs(self, docs):
        """Helper to convert LlamaIndex Documents to TextNodes."""
        nodes = []
        for doc in docs:
            # This is a fallback, metadata might be less rich
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
            # 1. Store the original, clean chunk text in metadata
            original_text = node.get_content()
            node.metadata['original_text'] = original_text

            # 2. Add file-level metadata
            node.metadata.update(self.current_file_metadata)

            # 3. Add hash
            node.metadata['hash'] = hashlib.md5(original_text.encode()).hexdigest()

            # 4. Build the new "searchable text"
            searchable_parts = [original_text]

            # --- MODIFIED: Dynamically build searchable content ---
            # Iterate through the "allow list" from config.yaml
            for key in self.embeddable_keys:
                if node.metadata.get(key):
                    value_str = str(node.metadata.get(key)).strip()
                    if value_str:
                        # Add a prefix for clarity, e.g., "Themes: ..."
                        searchable_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")
            # --- END MODIFICATION ---

            # 5. (Optional) Add summary based on *old* config flag (if present)
            # This is for backward compatibility if config.yaml isn't updated
            if self.cogarc_settings.get('include_summary_in_embedding', False):
                if "holistic_summary" in node.metadata:
                    searchable_parts.append(f"Summary: {node.metadata['holistic_summary']}")

            # 6. Set the node's main content to this new searchable string
            node.set_content("\n\n".join(searchable_parts))

            # 7. Set exclusions
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_key = list(node.metadata.keys())

            prepared_nodes.append(node)

        return prepared_nodes