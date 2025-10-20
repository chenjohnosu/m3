import click  # <-- ADDED
from core.ingestion.base_pipeline import BasePipeline
from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
from core.llm_manager import LLMManager


class CognitiveArchitectPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Cognitive Architect Pipeline...")
        llm_manager = LLMManager(config)
        self.cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        # --- NEW: Get embedding content flag from config ---
        self.include_summary_in_embedding = self.cogarc_settings.get('include_summary_in_embedding', False)
        click.echo(f"  > CogArc Pipeline: Include summary in embedding = {self.include_summary_in_embedding}", err=True)
        # --- End New ---

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

    def run(self, documents, doc_type):
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        # Start with the initial documents
        pipeline_data = {'documents': documents}

        if doc_type == 'interview':
            # Stage 0 is for interview stratification
            pipeline_data = self.stage_0.process(pipeline_data)
        else:
            print("Skipping Stage 0 (Q&A Stratification) for non-interview document.")

        if not pipeline_data.get('documents'):
            print("No content available for further processing.")
            return {}

        # The output of one stage becomes the input for the next
        structured_data = self.stage_1.process(pipeline_data)
        enriched_data = self.stage_2.process(structured_data)
        synthesized_data = self.stage_3.process(enriched_data)

        # --- MODIFIED: Apply final text formatting before returning ---
        final_nodes = synthesized_data.get("primary_nodes", [])
        if final_nodes:
            final_nodes = self._apply_and_prepare_nodes(final_nodes)
            synthesized_data["primary_nodes"] = final_nodes
        # --- END MODIFICATION ---

        print("--- Cognitive Architect Pipeline Finished ---")
        # Return the final data which contains the 'primary_nodes'
        return synthesized_data

    # --- ADDED THIS NEW HELPER METHOD ---
    def _apply_and_prepare_nodes(self, nodes: list):
        """
        Applies document-level metadata (like summary) to all nodes
        and constructs the final text to be embedded.
        """
        for node in nodes:
            # 1. Store the original, clean chunk text in metadata
            original_text = node.get_content()
            node.metadata['original_text'] = original_text

            # 2. Build the new "searchable text"
            #    Default: content + themes
            searchable_parts = [original_text]
            if "themes" in node.metadata:
                searchable_parts.append(f"Themes: {node.metadata['themes']}")

            # 3. (Optional) Add summary based on config
            if self.include_summary_in_embedding:
                if "holistic_summary" in node.metadata:
                    searchable_parts.append(f"Summary: {node.metadata['holistic_summary']}")

            # 4. Set the node's main content to this new searchable string
            #    This is what will be vectorized by the embedding model.
            node.set_content("\n\n".join(searchable_parts))

        return nodes