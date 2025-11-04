from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline
from core.llm_manager import LLMManager # Import LLMManager for type hinting

PIPELINES = {
    "cogarc": CognitiveArchitectPipeline,
}

def get_pipeline(pipeline_name, config, llm_manager: LLMManager): # <-- MODIFIED: Added llm_manager
    pipeline_class = PIPELINES.get(pipeline_name)
    if not pipeline_class:
        raise ValueError(f"Unknown pipeline: '{pipeline_name}'.")

    # --- MODIFIED ---
    # Pass the persistent llm_manager to the pipeline constructor
    return pipeline_class(config, llm_manager)
    # --- END MODIFIED ---