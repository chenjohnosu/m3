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
        cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        self.stage_0 = CogArcStage0Stratify(
            config, llm=llm_manager.get_llm(cogarc_settings['stage_0_model'])
        )
        self.stage_1 = CogArcStage1Structure(
            config, llm=llm_manager.get_llm(cogarc_settings['stage_1_model'])
        )
        self.stage_2 = CogArcStage2Enrich(
            config, llm=llm_manager.get_llm(cogarc_settings['stage_2_model'])
        )
        self.stage_3 = CogArcStage3Synthesis(
            config, llm=llm_manager.get_llm(cogarc_settings['stage_3_model'])
        )

    def run(self, documents, doc_type):
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        # Start with the initial documents
        pipeline_data = {'documents': documents}

        if doc_type == 'interview':
            # Stage 0 is a placeholder, for now, it just passes the documents through
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

        print("--- Cognitive Architect Pipeline Finished ---")
        # Return the final data which contains the 'primary_nodes'
        return synthesized_data